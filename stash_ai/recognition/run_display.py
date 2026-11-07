"""Small presentation state retained until command resources have closed."""
from collections import Counter
from time import monotonic

from .console_layout import time_line

from .reference_usage import usage_line


class RunDisplay:
    def __init__(self, report):
        self.report = report
        self.started = monotonic()
        self.counts = Counter()
        self.scans = Counter()
        self.rejections = {key: Counter() for key in ('face', 'body', 'person_evidence_unavailable')}
        self.references = {}
        self.rebuild = {}
        self.gpu = None
        self.debt = None
        self.selected = 0
        self.retention = 'skipped'
        self.recognition = 'failed'
        self.interrupted_cursor = None
        self.pose_resume = None
        self.run_id = None

    def finish(self, outcome):
        c = self.counts
        console = getattr(self.report, 'console', None)
        elapsed = time_line(monotonic() - self.started, console.phase_totals if console else {})
        lines = [f'Recognition {outcome}: recognition={self.recognition} retention={self.retention} · '
                 + elapsed.split(' | ', 1)[0] + (f' · run={self.run_id}' if self.run_id else '')]
        work = [f'scenes={self.selected}']
        if self.scans:
            work += [f'{kind}={self.scans[kind]}' for kind in ('fresh', 'resumed', 'reused') if self.scans[kind]]
            work.append(f'VTT complete={c["vtt_complete"]}/{self.selected} face_cues={c["face_cues"]} '
                        f'pose_cues={c["pose_cues"]} backfilled={c["pose_backfilled"]}')
        if c['scenes_processed']:
            work.append(f'published={c["scenes_processed"]-c["publications_unchanged"]} unchanged={c["publications_unchanged"]}')
        if 'portrait_people_selected' in c:
            work.append(f'portrait_people_selected={c["portrait_people_selected"]}')
        if self.rebuild:
            work.append('rebuild ' + ' '.join(f'{k}={v}' for k,v in self.rebuild.items()))
        lines.append('  ' + ' · '.join(work))
        issues = [f'{key}={c[key]}' for key in ('scenes_failed', 'publication_failed', 'publication_not_attempted',
                 'people_failed', 'metadata_failed', 'scene_links_removed', 'other_model_links_removed') if c[key]]
        if self.debt and (self.debt['pending_selected'] or self.debt['pending_outside_selection']):
            issues.append(f'pending selected={self.debt["pending_selected"]} outside={self.debt["pending_outside_selection"]}')
        for kind, reasons in self.rejections.items():
            if reasons:
                issues.append(kind + ' ' + ' '.join(f'{k}={n}' for k,n in reasons.most_common(2)))
        if self.pose_resume:
            scene, missing = self.pose_resume
            detail = 'coverage unavailable' if missing is None else (
                f'missing_cues={len(missing)} first_missing={min(missing) if missing else "none"}')
            issues.append(f'resume scene={scene} pass=pose {detail}')
        elif self.interrupted_cursor:
            scene, cursor = self.interrupted_cursor
            issues.append(f'resume scene={scene} committed_cues={cursor}')
        if issues:
            lines.append('  ' + ' · '.join(issues))
        details = [elapsed.partition(' | ')[2]] if ' | ' in elapsed else []
        if self.references:
            r = self.references
            details.append(f'face references={usage_line(r)} samples={r["samples"]} stage={r["stage"]}')
        if self.gpu is not None and self.gpu.peaks:
            from .live_resources import gib
            details.append('VRAM sampled peak ' + gib(self.gpu.peaks.get('process_bytes'))
                           + ' · torch reserved ' + gib(self.gpu.peaks.get('reserved_bytes')))
        if details:
            lines.append('  ' + ' | '.join(details))
        self.report.optional('recognition_summary', self.report.write, '\n'.join(lines))
