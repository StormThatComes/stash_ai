"""Linear marker overlap and semantic differences for model ablation."""
from stash_ai.evidence.marker_identity import is_occurrence


def _marker_key(marker):
    return (
        marker["tag"],
        round(float(marker["start"]), 3),
        None if marker["end"] is None else round(float(marker["end"]), 3),
        bool(is_occurrence(marker)),
        tuple(marker.get("extra_tags", ())),
    )


_OVERLAP_EPSILON = .0005


def _marker_group(marker):
    return marker["tag"], bool(is_occurrence(marker))


def _marker_end(marker):
    return float(marker["end"]) if marker["end"] is not None else float(marker["start"])


def _overlap_presence(markers, candidates):
    """Return overlap existence per marker in linear time for unioned components.

    union_intervals yields markers ordered by start; filtering that stream by
    label/type preserves the order required by the monotonic cursors below.
    """
    grouped = {}
    for candidate in candidates:
        grouped.setdefault(_marker_group(candidate), []).append(candidate)

    cursors = {}
    result = []
    for marker in markers:
        group = _marker_group(marker)
        rows = grouped.get(group, ())
        cursor = cursors.get(group, 0)
        start = float(marker["start"])
        end = _marker_end(marker)
        while cursor < len(rows) and _marker_end(rows[cursor]) + _OVERLAP_EPSILON < start:
            cursor += 1
        cursors[group] = cursor
        result.append(
            cursor < len(rows)
            and float(rows[cursor]["start"]) <= end + _OVERLAP_EPSILON
        )
    return result


def _marker_delta(full, reduced):
    reduced_keys = {_marker_key(item) for item in reduced}
    full_keys = {_marker_key(item) for item in full}
    full_overlap = _overlap_presence(full, reduced)
    reduced_overlap = _overlap_presence(reduced, full)
    return {
        "lost": sum(not overlap for overlap in full_overlap),
        "changed": sum(
            overlap and _marker_key(marker) not in reduced_keys
            for marker, overlap in zip(full, full_overlap)
        ),
        "added": sum(not overlap for overlap in reduced_overlap),
        "exact_removed": len(full_keys - reduced_keys),
        "exact_added": len(reduced_keys - full_keys),
        "occurrences_lost": sum(
            is_occurrence(marker) and not overlap
            for marker, overlap in zip(full, full_overlap)
        ),
    }


def _extras(markers):
    return {
        (
            item["tag"],
            round(float(item["start"]), 3),
            None if item["end"] is None else round(float(item["end"]), 3),
            bool(is_occurrence(item)),
            extra,
        )
        for item in markers for extra in item.get("extra_tags", ())
    }

