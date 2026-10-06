"""Validate init authorization independently of inference and Stash access."""
from stash_ai.storage.initialization import InitializationRun, string_list
from .common import tick


def check_initializations(db, errors, progress=None):
    runs = {}
    for scope, models, selection, scanned in db.execute('SELECT * FROM db_initializations'):
        tick(progress)
        try:
            runs[scope] = InitializationRun.decode(models, selection, scanned)
        except (TypeError, ValueError):
            errors['db_init_invalid'] += 1
    for scope, scene, owned, completed in db.execute('SELECT * FROM db_init_scenes'):
        tick(progress)
        try:
            string_list(owned)
            run = runs.get(scope)
            if (run is None or not isinstance(scene, str) or not scene.strip()
                    or type(completed) is not int or completed not in (0, 1)
                    or (run.selection is not None and scene not in run.selection)):
                raise ValueError('Invalid init scene.')
        except (TypeError, ValueError):
            errors['db_init_invalid'] += 1
