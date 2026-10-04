"""Compose pure interval unions with temporal extra-label qualification."""
from .interval_geometry import union_intervals as geometry_union
from .contracts import MarkerContribution, MergedMarker
from .labels import raw_label
from .temporal.extras import merged_extras


def union_intervals(contributions: list[MarkerContribution], *, borrow=False) -> list[MergedMarker]:
    components = geometry_union(contributions, borrow=borrow)
    for component in components:
        merged_extras(component)
        component['extra_tags'] = sorted({raw_label(label) for label in component['extra_tags']})
    return components
