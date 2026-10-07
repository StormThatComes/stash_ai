"""Shared ordering and caps for per-frame primary and extra detections."""


def eligible(tags, floor, exclusions=(), limit=0, *, validate=False):
    from .labels import raw_label
    values = []
    for tag, score in tags:
        if tag in exclusions:
            continue
        label = raw_label(tag) if validate else tag
        if score >= floor:
            values.append((label, score))
    values = sorted(values, key=lambda item: (-item[1], item[0]))
    return values[:limit] if limit else values


def selections(tags, high, floor, cores=(), tags_excluded=(), limit=0, *, validate=False):
    ordered = eligible(tags, min(high, floor), tags_excluded, validate=validate)
    primary = [(tag, score) for tag, score in ordered if score >= floor and tag not in cores]
    extras = [(tag, score) for tag, score in ordered if score >= high]
    return (primary[:limit], extras[:limit]) if limit else (primary, extras)
