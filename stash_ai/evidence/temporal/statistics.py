"""Sparse scene accumulation and qualification without a dense score matrix."""

CHECKPOINT_FORMAT = 1
ITEM_FIELDS = frozenset(('hits', 'high_hits', 'max_score', 'peak_time', 'seconds',
    'run_hits', 'run_start', 'last_end', 'last_frame', 'longest_hits', 'longest_seconds'))


class SceneAccumulator:
    """One small record per observed label, never a frame-by-vocabulary matrix."""
    def __init__(self, threshold, policy, state=None):
        self.threshold, self.policy = threshold, policy
        if state is None:
            self.items, self.frames, self.covered, self.last_end = {}, 0, 0.0, 0.0
        else:
            if (not isinstance(state, dict) or set(state) != {'format', 'items', 'frames', 'covered', 'last_end'}
                    or type(state['format']) is not int or state['format'] != CHECKPOINT_FORMAT
                    or not isinstance(state['items'], dict)
                    or any(not isinstance(item, dict) or set(item) != ITEM_FIELDS for item in state['items'].values())):
                raise ValueError('Unsupported or incomplete scene checkpoint analysis; use --restart.')
            self.items = {int(k): dict(v) for k, v in state['items'].items()}
            self.frames, self.covered, self.last_end = state['frames'], state['covered'], state['last_end']

    def add(self, scores, cues):
        import numpy as np
        floor = self.threshold * self.policy.scene_support_ratio
        for row, cue in zip(scores, cues):
            values = ((int(index), float(row[index])) for index in np.flatnonzero(row >= floor))
            self.add_sparse(values, cue)

    def add_sparse(self, values, cue):
        """Consume retained observations without building a vocabulary-sized row."""
        floor = self.threshold * self.policy.scene_support_ratio
        self.frames += 1
        self.covered += max(0.0, cue.end - max(cue.start, self.last_end))
        self.last_end = max(self.last_end, cue.end)
        for index, score in values:
            if score < floor:
                continue
            item = self.items.setdefault(index, dict(hits=0, high_hits=0, max_score=0.0,
                seconds=0.0, run_hits=0, run_start=cue.start, last_end=cue.start,
                longest_hits=0, longest_seconds=0.0, last_frame=0, peak_time=cue.start))
            if (not item["hits"] or item["last_frame"] != self.frames - 1
                    or cue.start > item["last_end"] + 0.001):
                item["run_hits"], item["run_start"] = 0, cue.start
            item["hits"] += 1
            item["high_hits"] += int(score >= self.threshold)
            # Keep an actual sampled time, including detections below the
            # marker retention floor. Equal maxima retain the earliest time.
            if score > item["max_score"]:
                item["peak_time"] = cue.start
            item["max_score"] = max(item["max_score"], score)
            item["seconds"] += max(0.0, cue.end - max(cue.start, item["last_end"]))
            item["last_end"] = max(item["last_end"], cue.end)
            item["run_hits"] += 1
            item["last_frame"] = self.frames
            item["longest_hits"] = max(item["longest_hits"], item["run_hits"])
            item["longest_seconds"] = max(item["longest_seconds"], item["last_end"] - item["run_start"])

    def state(self):
        # A checkpoint must not share mutable counters with an unfinished batch.
        return dict(format=CHECKPOINT_FORMAT, items={k: dict(v) for k, v in self.items.items()}, frames=self.frames,
                    covered=self.covered, last_end=self.last_end)

    def statistics(self, labels):
        return {labels[index]: dict(hits=item["hits"], high_hits=item["high_hits"], max_score=item["max_score"],
            sampled_frames=self.frames, sampled_seconds=self.covered, support_seconds=item["seconds"],
            sample_fraction=item["hits"] / self.frames if self.frames else 0.0,
            coverage_fraction=min(1.0, item["seconds"] / self.covered) if self.covered else 0.0,
            longest_hits=item["longest_hits"], longest_seconds=item["longest_seconds"],
            peak_time=item["peak_time"])
            for index, item in self.items.items()}


def scene_reason(score, stats, threshold, policy):
    if stats is None:
        return None
    if stats["max_score"] >= threshold + (1.0 - threshold) * policy.scene_strong_margin:
        return "strong_peak"
    if score >= threshold and stats["high_hits"] >= policy.scene_min_hits:
        return "repeated_high"
    if (stats["hits"] >= policy.scene_repeat_hits and stats["longest_hits"] >= policy.scene_repeat_run
            and stats["sample_fraction"] >= policy.scene_repeat_fraction):
        return "persistent_support"
    return None
