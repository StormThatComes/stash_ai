"""Run counters, bounded issues and user-facing progress coordination."""
from collections import Counter
import sys
from .timing import RunTimings
from .summary import write_summary


class RunReport:
    """Fixed-size counters and bounded error examples, even for many scenes."""
    heartbeat_seconds = 5.0
    def __init__(self, verbose=False, write=print, max_groups=6, examples=2):
        self.verbose, self.write = verbose, write
        self.max_groups, self.examples = max_groups, examples
        self.groups = {}
        self.overflow = 0
        self.counts = Counter()
        self._timing = RunTimings(self)
        self.timings, self.timing_calls = self._timing.timings, self._timing.timing_calls
        self.threshold_preflight = None
        self.diagnostics = None
        self.unread_scenes = {}
        self.dry_run_plan = None
        self.active_requests = {}
        self.profile = None
        self.issue_counts = Counter()
        self.count_sources = []
        self.history = None
        self.workload = dict(verified_input_pairs=0, verified_cues=0, input_fingerprint=None)
        self._input_digest = 0

    def record_input_workload(self, scene, model, cues, hashes, *, snapshot=None):
        if self.profile is None:
            return
        from stash_ai.inference.input_identity import cue_manifest, frame_signature
        from stash_ai.signatures import digest
        fingerprint = snapshot.frame_signature if snapshot is not None else frame_signature(cue_manifest(cues), hashes)
        self._input_digest = (self._input_digest + int(digest([str(scene), model, fingerprint]), 16)) % (1 << 256)
        self.workload['verified_input_pairs'] += 1
        self.workload['verified_cues'] += len(cues)
        self.workload['input_fingerprint'] = digest(hex(self._input_digest))

    def record_progress(self):
        progress = getattr(self.history, 'progress', None)
        if progress is not None:
            self.optional('history_progress', progress)

    def optional(self, name, callback, *args, **kwargs):
        """Count optional reporting failures without changing the operation."""
        primary = sys.exception()
        try:
            return callback(*args, **kwargs)
        except Exception as error:
            key = name + '_failures'
            with self._timing._timing_lock:
                self.counts[key] += 1
                first = self.counts[key] == 1
            message = f'Optional {name} failed ({type(error).__name__}).'
            if primary is not None:
                primary.add_note(message)
            if first:
                try:
                    self.write(message)
                except Exception:
                    pass

    def reuse_decision(self, scope, scene, model, action, status, reasons, details):
        if status != 'started':
            self.counts['reuse_decision_' + action] += 1
            for reason in reasons:
                self.counts['reuse_reason_' + reason] += 1
        self.detail(f'Reuse {model}/scene {scene}: {action}/{status}; ' + ', '.join(reasons))
        if self.history is not None:
            self.history.call('record_decision', self.history.identifier, scope, scene, model,
                              action, status, reasons, details)

    def measure(self, stage, *, model=None, work=None, units=0, unit=''):
        return self._timing.measure(stage, model=model, work=work, units=units, unit=unit)

    def work(self, name=None):
        if name is not None:
            self._timing._timers.work = name
        return getattr(self._timing._timers, 'work', '')

    def model(self, name):
        return self._timing.model(name)

    def watch_counts(self, source):
        if not any(item is source for item in self.count_sources):
            self.count_sources.append(source)

    def model_outcome(self, model, name):
        self.work(name)
        if self.profile is not None:
            self.optional('profile_outcome', self.profile.outcome, model, name)

    def record_model(self, config, *, identity=None, tagger=None):
        try:
            self._record_model(config, identity=identity, tagger=tagger)
        except Exception:
            # Optional backend metadata cannot turn a loaded tagger into a failure.
            pass

    def _record_model(self, config, *, identity=None, tagger=None):
        if self.profile is None:
            return
        source = config.source
        row = self.profile.models.setdefault(source.key, {})
        row.update(repo=source.repo, revision=source.revision, scene_threshold=config.scene_threshold,
                   marker_threshold=config.marker_threshold, batch_size=config.batch_size, device=config.device,
                   scene_cache_floor=config.scene_cache_floor, marker_cache_floor=config.marker_cache_floor)
        if identity is not None:
            row.update(vocabulary=identity['vocabulary'], asset_hashes=[value for _, value in identity['files']])
        row.setdefault('execution_observed', False)
        if tagger is not None:
            row['execution_observed'] = True
            row['effective_batch_size'] = tagger.batch_size
            if hasattr(tagger, 'device'):
                row['effective_device'] = str(tagger.device)
                import sys
                torch = sys.modules.get('torch')
                if torch is not None and str(tagger.device).startswith('cuda'):
                    try:
                        row['gpu'] = torch.cuda.get_device_name(tagger.device)
                    except Exception as exc:
                        row['gpu_metadata_error'] = type(exc).__name__
            session = getattr(tagger, 'session', None)
            if session is not None and hasattr(session, 'get_providers'):
                row['providers'] = session.get_providers()

    def request(self, operation, read):
        return self._timing.request(operation, read)

    def bind_read(self, operation):
        return self._timing.bind(operation)

    def response_size(self, size):
        self._timing.response_size(size)

    def record_counts(self, **values):
        with self._timing._timing_lock:
            self.counts.update(values)
        if self.diagnostics is not None:
            self.optional('diagnostics_counts', self.diagnostics.record_counts, values)

    def detail(self, message):
        if self.verbose:
            self.write(message)

    def issue(self, stage, source, scene, exc, message):
        self.issue_counts['/'.join((stage, source, type(exc).__name__))] += 1
        key = (stage, source, type(exc).__name__)
        if key not in self.groups:
            if len(self.groups) >= self.max_groups:
                self.overflow += 1
                if self.verbose:
                    self.write(f'ERROR {stage} {source} scene {scene}: {message}')
                return
            self.groups[key] = dict(count=0, examples=[])
        group = self.groups[key]
        group['count'] += 1
        example = f"scene {scene}: {message}" if scene is not None else message
        example = example.replace('\n', ' ')[:350]
        if len(group['examples']) < self.examples:
            group['examples'].append(example)
        if self.verbose or group['count'] == 1:
            self.write(f"ERROR {stage} {source}: {example}")

    def progress(self, bar, counts):
        bar.set_postfix(processed=counts['processed'], reused=counts['reused'],
                        skipped=counts['skipped'], errors=counts['errors'], refresh=False)

    def record_threshold_preflight(self, checked, failed, reasons=None, applied=0, *,
                                   selected=None, skipped_no_evidence=0):
        """Keep one compact summary for an atomic threshold update attempt."""
        checked, skipped = int(checked), int(skipped_no_evidence)
        self.threshold_preflight = dict(
            selected=int(checked + skipped if selected is None else selected),
            checked=checked, skipped_no_evidence=skipped,
            failed=int(failed), applied=int(applied),
            reasons=dict(reasons or {}),
        )

    def finish(self, title, counts, cache=None):
        self.watch_counts(counts)
        if self.profile is not None and cache is not None:
            self.counts.update({f'ram_cache_{name}': value for name, value in cache.stats.items()})
        write_summary(self, title, counts, cache)
