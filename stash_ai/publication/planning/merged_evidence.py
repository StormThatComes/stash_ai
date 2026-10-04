"""Merged publication evidence."""
from __future__ import annotations
from stash_ai.publication import markers as marker_ops
from stash_ai.publication.index import MarkerIntervalIndex, PreviousMarkerRanges
import stash_ai.evidence.intervals as merge_intervals
import stash_ai.evidence.labels as merge_labels
from stash_ai.runtime.timing import stage
from .types import PreparedEvidence


def _previous_ranges(app, store, scene_id):
    if not app.CREATE_MARKERS or app.MARKER_USE_END_TIME:
        return []
    from stash_ai.storage.recovery import publications
    result = []
    for snapshot in (*store.snapshot(scene_id, "markers", borrow=True),
                     *publications(store, scene_id, 'markers')):
        if snapshot:
            result.extend(merge_intervals.union_intervals(snapshot.get("contributions", []), borrow=True))
    return result


def prepare_sync_evidence(app, stash, store, scene, scene_hits, spans, *, cleanup, report):
    """Build merged markers/scene labels and apply cleanup-only retention rules."""
    scene_id = str(scene["id"])
    previous_ranges = PreviousMarkerRanges(_previous_ranges(app, store, scene_id))
    with stage(report, 'marker_union'):
        markers = merge_intervals.union_intervals(spans, borrow=True) if app.CREATE_MARKERS else []
    if report is not None:
        report.record_counts(marker_contributions=len(spans), union_markers=len(markers),
                             scene_contributions=len(scene_hits))
    scene_labels = {merge_labels.raw_label(hit["tag"]) for hit in scene_hits} if app.APPLY_SCENE_TAGS else set()
    if cleanup:
        # Cleanup never restores outputs that the user removed since the last sync.
        scene_labels.intersection_update(tag["name"][len(merge_labels.AI.prefix):] for tag in scene.get("tags", [])
                                         if tag["name"].startswith(merge_labels.AI.prefix))
        scene_hits = [hit for hit in scene_hits if hit["tag"] in scene_labels]
        present = [marker for marker in scene.get("scene_markers", [])
                   if marker_ops.owned_marker(marker, merge_labels.AI)]
        existing = MarkerIntervalIndex(present if markers else [], previous_ranges, by_name=True)
        markers = [marker for marker in markers if existing.overlapping(
            merge_labels.AI.prefix + marker["tag"], {"seconds": marker["start"], "end_seconds": marker["end"]})]
        spans = [item for marker in markers for item in marker["contributors"]]
    return PreparedEvidence(scene_hits, spans, markers, scene_labels, previous_ranges)

