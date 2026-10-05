"""Explicit qualification rules for historical aggregates and imported extras."""


def aggregate_reason(score, threshold):
    return 'aggregate' if score is None or score >= threshold else None


def plain_extras(record):
    """Only historical records may carry unproved plain additional labels."""
    from stash_ai.evidence.formats import current
    if current(record.get('meta', {})):
        raise ValueError('Current marker contributions require temporal extra_support evidence.')
    return record.get('extra_tags', ())
