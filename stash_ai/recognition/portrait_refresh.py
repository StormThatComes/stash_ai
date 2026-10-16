"""Select existing portrait targets without recognition or Stash publication."""
from collections.abc import Mapping

from stash_ai.runtime.console import activity
from .database import encoded
from .models import namespace, ensure_assets, asset_selection
from .progress import tracking


class LazyAssets(Mapping):
    """Prepare only model assets actually consumed by this invocation."""
    def __init__(self, config, factory, report):
        self.config, self.factory, self.report = config, factory, report
        self.selective = factory is ensure_assets
        self.paths = {} if self.selective else None
        self.errors = {}

    def _load(self, keys=None):
        group = tuple(keys) if self.selective else ('all',)
        if group in self.errors:
            raise ValueError(self.errors[group])
        needed = self.paths is None or self.selective and any(key not in self.paths for key in group)
        if needed:
            try:
                with activity(self.report, 'prepare required recognition assets'):
                    if self.selective:
                        self.paths.update(self.factory(self.config, write=self.report.write, keys=group))
                    else:
                        self.paths = self.factory(self.config, write=self.report.write)
            except Exception as error:
                from .diagnostics import message
                self.errors[group] = message(error)[:500]
                raise
        return self.paths

    def prepare_identity(self):
        models = self.config.data['models']
        return self._load((models['detector'], models['recognizer']))

    def __getitem__(self, key):
        if key in ('expression_mesh', 'expression_blendshapes'):
            keys = ('expression_mesh', 'expression_blendshapes')
        elif key in (self.config.data['models']['detector'], self.config.data['models']['recognizer']):
            models = self.config.data['models']
            keys = (models['detector'], models['recognizer'])
        else:
            keys = (key,)
        return self._load(keys)[key]

    def __iter__(self):
        return iter(asset_selection(self.config)[0] if self.selective else self._load())

    def __len__(self):
        return len(asset_selection(self.config)[0] if self.selective else self._load())


def retained_targets(database, config, scope, scene_ids=None, *, performer_ids=None):
    """Current memberships only; never resurrect historical candidate decisions.

    The normal canonical-target pass additionally excludes roots with unresolved
    performer merges. Model namespaces and Stash instances cannot be mixed.
    Performer and scene filters intersect; performer selection alone retains all
    of each selected person's eligible evidence.
    """
    scenes = None if scene_ids is None else encoded(scene_ids)
    performers = None if performer_ids is None else encoded(performer_ids)
    rows = database.connection.execute('''SELECT DISTINCT p.id,p.performer,s.scene,s.id AS scan
      FROM people p JOIN members m ON m.person=p.id
      JOIN observations o ON o.id=m.observation JOIN scans s ON s.id=o.scan
      WHERE p.scope=? AND p.namespace=? AND p.performer IS NOT NULL
        AND s.scope=p.scope AND s.namespace=p.namespace AND s.status='complete'
        AND o.rejected IS NULL
        AND NOT EXISTS (SELECT 1 FROM identity_merges a WHERE a.source=p.id)
        AND (? IS NULL OR s.scene IN (SELECT value FROM json_each(?)))
        AND (? IS NULL OR p.performer IN (SELECT value FROM json_each(?)))
      ORDER BY p.id,s.scene,s.id''',
      (scope, namespace(config), scenes, scenes, performers, performers))
    ready = {}
    with tracking(rows, 'Read retained portrait identities', unit='assignment', position=0) as assignments:
        for row in assignments:
            target = ready.setdefault(row['id'], {'person': row['id'], 'performer': row['performer'], 'scans': []})
            target['scans'].append({'scene': row['scene'], 'scan': row['scan']})
    return ready
