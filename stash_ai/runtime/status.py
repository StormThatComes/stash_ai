"""The same failure classification for history, profiles and concrete plans."""


def completion_status(exception_type=None, *, errors=False):
    if exception_type is not None:
        return 'error' if issubclass(exception_type, Exception) else 'interrupted'
    return 'error' if errors else 'finished'
