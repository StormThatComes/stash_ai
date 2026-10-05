"""Explicit qualification rules for historical aggregates and imported extras."""


def aggregate_reason(score, threshold):
    return 'aggregate' if score is None or score >= threshold else None
