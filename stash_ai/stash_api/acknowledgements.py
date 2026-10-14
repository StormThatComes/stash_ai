"""Validate mutation results before publication observers confirm a write."""


def entity(value, expected=None):
    identifier = value.get('id') if isinstance(value, dict) else None
    if (not isinstance(identifier, str) or not identifier.isascii() or not identifier.isdecimal()
            or int(identifier) <= 0 or str(int(identifier)) != identifier
            or expected is not None and identifier != str(expected)):
        raise RuntimeError('Stash mutation acknowledgement ID was not confirmed; outcome is unknown.')
    return value


def boolean(value):
    if value is not True:
        raise RuntimeError('Stash mutation acknowledgement was not confirmed; outcome is unknown.')
