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
        parts = [f'recognition={self.recognition} retention={self.retention}',
                 f'scenes={self.selected} fresh={self.scans["fresh"]} resumed={self.scans["resumed"]} '
                 f'reused={self.scans["reused"]} scenes_failed={c["scenes_failed"]}',
                 f'published={c["scenes_processed"] - c["publications_unchanged"]} '
                 f'unchanged={c["publications_unchanged"]}']
        if self.run_id is not None:
            parts.append(f'run={self.run_id}')
        if self.scans:
            parts.append(f'VTT complete={c["vtt_complete"]}/{self.selected} · face cues={c["face_cues"]}'
                         f' · pose cues={c["pose_cues"]} (backfilled={c["pose_backfilled"]})')
        if self.pose_resume is not None:
            scene, missing = self.pose_resume
            detail = 'durable coverage unavailable' if missing is None else (
                f'missing_cues={len(missing)} first_missing={min(missing) if missing else "none"}')
            parts.append(f'resume scene={scene} pass=pose {detail}')
        if 'portrait_people_selected' in c:
            parts.append(f'portrait_people_selected={c["portrait_people_selected"]}')
        if self.rebuild:
            parts.append('rebuild ' + ' '.join(f'{k}={v}' for k, v in self.rebuild.items()))
        if self.debt is not None:
            parts.append(f'pending selected={self.debt["pending_selected"]} outside={self.debt["pending_outside_selection"]}')
        for name in ('publication_failed', 'publication_not_attempted', 'people_failed', 'metadata_failed', 'scene_links_removed', 'other_model_links_removed'):
            if c[name]:
                parts.append(f'{name}={c[name]}')
        for kind, reasons in self.rejections.items():
            if reasons:
                parts.append(kind + ' ' + ' '.join(f'{k}={n}' for k, n in reasons.most_common(2)))
        if self.references:
            r = self.references
            parts.append(f'face references={usage_line(r)} samples={r["samples"]} stage={r["stage"]}')
        if self.gpu is not None and self.gpu.peaks:
            from .live_resources import gib
            peak = self.gpu.peaks
            parts.append(f'VRAM sampled peak: process {gib(peak.get("process_bytes"))}, '
                         f'torch reserved {gib(peak.get("reserved_bytes"))}')
        if self.interrupted_cursor is not None:
            scene, cursor = self.interrupted_cursor
            parts.append(f'resume scene={scene} committed_cues={cursor}')
        console = getattr(self.report, 'console', None)
        parts.append(time_line(monotonic() - self.started, console.phase_totals if console else {}))
        self.report.optional('recognition_summary', self.report.write,
                             f'Recognition {outcome}: ' + ' | '.join(parts))
