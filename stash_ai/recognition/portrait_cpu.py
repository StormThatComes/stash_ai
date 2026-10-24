"""Pure portrait CPU jobs: no SQLite, HTTP, model sessions or publication."""


def identity_batch(payload):
    rows, groups, person, config, cancel = payload
    from .clustering import vector
    from .portraits import identity_reason
    from .scoring import score_scope
    result = []
    from stash_ai.runtime.cancellation import use
    with score_scope(), use(cancel):
        for row in rows:
            cancel.check()
            try:
                result.append(identity_reason(vector(row), groups, person, config))
            except (ValueError, TypeError):
                result.append(None)  # Coordinator retains the original error boundary.
    return result


def visibility_input(image, face):
    from .visibility import prepare, VisibilityGeometryError
    try:
        return prepare(image, face)
    except VisibilityGeometryError as error:
        return error
    finally:
        image.close()
