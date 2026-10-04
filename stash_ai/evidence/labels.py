"""Merged namespace and validated raw labels, independent of orchestration."""
from stash_ai.models import ModelSource, SOURCES
from stash_ai.evidence.marker_identity import OCCURRENCE_TAG


AI = ModelSource("ai", "AI", "ai:", "ai:marker:stash-taggers", "[AI] ", "[AI hierarchy v1]")


def raw_label(value):
    if (not isinstance(value, str) or not value or value != value.strip() or "\n" in value or "\r" in value
            or AI.prefix + value in (AI.owner_tag, OCCURRENCE_TAG) or value.startswith("category:")):
        raise ValueError(f"Invalid or reserved merged label: {value!r}")
    return value


def source_codes(sources):
    return [source.prefix.rstrip(":") for key, source in SOURCES.items() if key in sources]

