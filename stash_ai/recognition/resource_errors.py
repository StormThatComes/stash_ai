"""Typed admission failure: a trial may be skipped without hiding data errors."""


class ResourceLimit(ValueError):
    """A configured replay or matching resource budget cannot admit this work."""
