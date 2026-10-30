"""Typed admission failure: a trial may be skipped without hiding data errors."""


class ResourceLimit(ValueError):
    """A configured replay or matching resource budget cannot admit this work."""

    def __init__(self, message, *, budget=None, limit=None, requested=None):
        super().__init__(message)
        self.budget, self.limit, self.requested = budget, limit, requested

    def document(self):
        return dict(kind='configured_resource_limit',budget=self.budget,limit=self.limit,requested=self.requested)
