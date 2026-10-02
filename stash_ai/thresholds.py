"""Marker publication thresholds over immutable, bounded inference evidence."""

import math
from collections import Counter
from types import SimpleNamespace

from tqdm import tqdm

OVERRIDE = 'marker_threshold_override'


def indexed_signature(segment, inference=None):
    from stash_ai.progress import digest
    signature = segment['meta'].get('inference_signature') if inference is None else inference
    from stash_ai.scene_cache import SCENE_OVERRIDE
    if SCENE_OVERRIDE in segment and OVERRIDE in segment:
        return digest([signature, {
            'scene_threshold': segment[SCENE_OVERRIDE],
            'marker_threshold': segment[OVERRIDE],
        }])
    if SCENE_OVERRIDE in segment:
        return digest([signature, {'scene_threshold': segment[SCENE_OVERRIDE]}])
    return digest([signature, segment[OVERRIDE]]) if OVERRIDE in segment else signature


def marker_context(app, records):
    """Freeze effective thresholds with each publication, including cleanup replay."""
    if hasattr(app, 'MARKER_THRESHOLDS'):
        return app
    from stash_ai.merge import marker_segment_signature
    context = SimpleNamespace(**vars(app))
    context.MARKER_THRESHOLDS = {
        marker_segment_signature(segment): segment[OVERRIDE]
        for segments in records.values() for segment in segments if OVERRIDE in segment}
    return context


def marker_threshold(segment, thresholds=None):
    from stash_ai.merge import marker_segment_signature
    if thresholds is not None:
        return thresholds.get(marker_segment_signature(segment), segment['meta'].get('marker_threshold'))
    return segment.get(OVERRIDE, segment['meta'].get('marker_threshold'))


def _score(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def threshold_failure_reason(exc):
    """Collapse scene-specific preflight errors into stable operator-facing reasons."""
    message = str(exc).strip()
    if ': ' in message:
        prefix, remainder = message.split(': ', 1)
        if '/' in prefix:
            message = remainder
    lower = message.lower()
    rules = (
        ('no scene results', 'missing scene results'),
        ('scene frame evidence is missing', 'missing scene frame evidence'),
        ('scene cache floor', 'scene cache floor too high'),
        ('no cached marker frames', 'missing marker frames'),
        ('imported ranges cannot be rethresholded', 'imported marker ranges'),
        ('marker cache floor', 'marker cache floor too high'),
        ('cache floor', 'marker cache floor too high'),
        ('invalid cached marker threshold', 'invalid cached marker threshold'),
        ('marker blocklist metadata', 'unknown marker blocklist'),
        ('unblocked marker tags', 'unblocked marker tags require inference'),
        ('scene_threshold changed', 'scene threshold requires inference'),
        ('finish the pending merge', 'pending publication'),
    )
    return next((reason for token, reason in rules if token in lower), type(exc).__name__)


def validate_marker_blocklist(app, segment, identity, *, allow_unknown=False):
    """Previously filtered labels cannot be recovered by changing publication policy."""
    if segment['mode'] != 'frames' or segment['meta'].get('origin') != 'inference':
        return
    if allow_unknown and 'marker_tag_blocklist' not in segment['meta']:
        return
    retained = segment['meta'].get('marker_tag_blocklist')
    if not isinstance(retained, list) or any(not isinstance(tag, str) for tag in retained):
        raise ValueError(f'{identity}: marker blocklist metadata is missing or invalid; '
                         'run inference with --restart for this scene/model.')
    unblocked = sorted(set(retained) - app.MARKER_TAG_BLOCKLIST)
    if unblocked:
        raise ValueError(f'{identity}: unblocked marker tags {unblocked!r} are absent from cached frames; '
                         'run inference with --restart for this scene/model.')


def validate_marker_cache(app, records, *, allow_unknown_blocklist=False):
    """Check retained coverage before cached frames can build markers or votes."""
    from stash_ai.temporal import marker_floor
    for model, segments in records.items():
        for segment in segments:
            meta = segment['meta']
            validate_marker_blocklist(app, segment, model, allow_unknown=allow_unknown_blocklist)
            # Imported spans and old frames without retention metadata keep
            # their existing semantics; no original support floor can be proven.
            if segment['mode'] != 'frames' or 'marker_score_floor' not in meta:
                continue
            threshold = marker_threshold(segment, getattr(app, 'MARKER_THRESHOLDS', None))
            if not _score(threshold):
                raise ValueError(f'{model}: invalid cached marker threshold; run inference.')
            needed = marker_floor(app.TEMPORAL_POLICY, threshold)
            floor = meta['marker_score_floor']
            if not _score(floor) or floor > needed:
                raise ValueError(f'{model}: marker cache floor {floor!r} exceeds required {needed:g}; run inference.')


def _model_records(store, scene, kind, model, *, direct=False, metadata_only=False):
    if metadata_only:
        return store.result_metadata(scene, kind, model)
    if direct:
        return store.model_records(scene, kind, model, borrow=True)
    return store.records(scene, kind, model, borrow=True).get(model, [])


def threshold_targets(store, scenes, settings, kinds):
    """Select requested output kinds using result keys/indexes without decoding payloads."""
    scene_ids = [str(scene) for scene in scenes]
    models = [config.source.key for config in settings]
    requested = frozenset(kinds)
    known = store.result_presence(models) | store.result_index_presence(models)
    targets = []
    for scene in scene_ids:
        for config in settings:
            model = config.source.key
            available = frozenset(kind for kind in requested if (scene, model, kind) in known)
            if available:
                targets.append((scene, config, available))
    selected = len(scene_ids) * len(settings)
    return targets, selected - len(targets), selected


def validate(app, store, scene, config, *, check_scene=True, direct=False, metadata_only=False):
    """Reject incomplete evidence; a missing detection is never treated as zero."""
    from stash_ai.temporal import marker_floor
    model = config.source.key
    segments = _model_records(store, scene, 'markers', model, direct=direct, metadata_only=metadata_only)
    if not segments:
        raise ValueError(f'{scene}/{model}: no cached marker frames; run inference for this scene/model.')
    from stash_ai.scene_cache import scene_threshold
    scene_segments = (_model_records(store, scene, 'scene', model, direct=direct, metadata_only=metadata_only)
                      if check_scene else [])
    for segment in scene_segments:
        if segment['meta'].get('origin') == 'inference' and scene_threshold(segment) != config.scene_threshold:
            raise ValueError(f'{scene}/{model}: scene_threshold changed; scene statistics require inference.')
    needed = marker_floor(app.TEMPORAL_POLICY, config.marker_threshold)
    for segment in segments:
        meta = segment['meta']
        if segment['mode'] != 'frames' or meta.get('origin') != 'inference':
            raise ValueError(f'{scene}/{model}: imported ranges cannot be rethresholded; run inference.')
        validate_marker_blocklist(app, segment, f'{scene}/{model}')
        # Before temporal retention was introduced, only high detections survived.
        floor = meta.get('marker_score_floor', meta.get('marker_threshold'))
        if not _score(floor) or floor > needed:
            raise ValueError(f'{scene}/{model}: cache floor {floor!r} exceeds required {needed:g}; run inference.')
        if check_scene and not scene_segments and meta.get('scene_threshold') != config.scene_threshold:
            raise ValueError(f'{scene}/{model}: scene_threshold changed; scene statistics require inference.')
    return segments


def apply(app, store, scene, config, segments=None, *, direct=False, pending_checked=False,
          metadata_only=False):
    """Store policy separately; retain every original frame, score and signature."""
    segments = validate(app, store, scene, config, direct=direct,
                        metadata_only=metadata_only) if segments is None else segments
    if all(marker_threshold(s) == config.marker_threshold for s in segments):
        return False
    if not pending_checked and store.has_pending_publication(scene):
        raise RuntimeError(f'{scene}: finish the pending merge or use --abandon-pending before changing thresholds.')
    updated = []
    for segment in segments:
        value = dict(segment)
        value.pop(OVERRIDE, None)
        if config.marker_threshold != segment['meta']['marker_threshold']:
            value[OVERRIDE] = config.marker_threshold
        updated.append(value)
    with store.connection:
        store.set_result_threshold(scene, config.source.key, 'markers', config.marker_threshold, updated,
                                   metadata_only=metadata_only)
    return True


def reuse(app, store, scene, config, identity, cues, hashes, kinds):
    """Prove the original inference matches, allowing only a marker threshold edit."""
    from stash_ai.progress import completed, digest
    from stash_ai.scene_cache import reuse as reuse_scenes
    if reuse_scenes(app, store, scene, config, identity, cues, hashes, kinds):
        return True
    if not app.CREATE_MARKERS:
        return False
    try:
        segments = validate(app, store, scene, config)
    except ValueError:
        return False
    original = segments[-1]['meta'].get('marker_threshold')
    signature = digest({'model': {**identity, 'marker_threshold': original},
        'cues': [[c.start, c.end, c.image, list(c.rect)] for c in cues], 'media': hashes})
    for kind in kinds:
        expected = indexed_signature(segments[-1], signature) if kind == 'markers' else signature
        if not completed(store, scene, config.source.key, [kind], expected):
            return False
    apply(app, store, scene, config, segments)
    return True


def rethreshold(app, store, scenes, settings, report):
    """Preflight only saved evidence; fully absent scene/model pairs are a no-op."""
    targets, skipped, selected = threshold_targets(store, scenes, settings, ('markers',))
    errors = 0
    reasons = Counter()
    pending_scenes = store.pending_publication_scenes()
    from stash_ai.state_metadata import backfill_batch
    with backfill_batch(store):
        for scene, config, _ in tqdm(targets, desc='Check marker thresholds', unit='pair'):
            try:
                validate(app, store, scene, config, direct=True, metadata_only=True)
                if str(scene) in pending_scenes:
                    raise RuntimeError('Finish the pending merge or use --abandon-pending first.')
            except (ValueError, RuntimeError) as exc:
                report.issue('marker thresholds', config.source.key, scene, exc, str(exc))
                reasons[threshold_failure_reason(exc)] += 1
                errors += 1
    if errors:
        report.record_threshold_preflight(
            len(targets), errors, reasons, applied=0,
            selected=selected, skipped_no_evidence=skipped)
        return dict(errors=errors)
    changed = 0
    for scene, config, _ in tqdm(targets, desc='Apply marker thresholds', unit='pair'):
        changed += apply(app, store, scene, config, direct=True, pending_checked=True, metadata_only=True)
    report.record_threshold_preflight(
        len(targets), 0, applied=changed,
        selected=selected, skipped_no_evidence=skipped)
    return dict(marker_thresholds_updated=changed)
