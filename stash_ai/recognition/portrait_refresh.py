"""Select existing portrait targets without recognition or Stash publication."""
from collections.abc import Mapping

from stash_ai.runtime.console import activity
from .database import encoded
from .models import namespace, ensure_assets, asset_selection, identity_asset_keys
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
        return self._load(identity_asset_keys(self.config))

    def __getitem__(self, key):
        if key in ('expression_mesh', 'expression_blendshapes'):
            keys = ('expression_mesh', 'expression_blendshapes')
        elif key in (self.config.data['models']['detector'], self.config.data['models']['recognizer']):
            models = self.config.data['models']
            keys = identity_asset_keys(self.config)
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
    performers = None if performer_ids is None else encoded(performer_ids)
    ready = {}
    if scene_ids is not None:
        scene_ids=list(dict.fromkeys(scene_ids))
    selections = [None] if scene_ids is None else [scene_ids[i:i+256] for i in range(0,len(scene_ids),256)]
    for selection in selections:
        source = ('people p JOIN members m ON m.person=p.id JOIN observations o ON o.id=m.observation '
                  'JOIN scans s ON s.id=o.scan')
        parameters=[scope,namespace(config)]
        selected_filter=''
        if selection is not None:
            source = ('json_each(?) chosen CROSS JOIN scans s ON s.scene=chosen.value AND s.scope=? AND s.namespace=? '
                      'JOIN observations o ON o.scan=s.id JOIN members m ON m.observation=o.id JOIN people p ON p.id=m.person')
            parameters=[encoded(selection),scope,namespace(config),*parameters]
        if performers is not None:
            selected_filter=' AND p.performer IN (SELECT value FROM json_each(?))'
            parameters.append(performers)
        rows = database.connection.execute('SELECT DISTINCT p.id,p.performer,s.scene,s.id AS scan FROM '+source+
            " WHERE p.scope=? AND p.namespace=? AND p.performer IS NOT NULL AND s.scope=p.scope "
            "AND s.namespace=p.namespace AND s.status='complete' AND o.rejected IS NULL "
            'AND NOT EXISTS (SELECT 1 FROM identity_merges a WHERE a.source=p.id)'+selected_filter+
            ' ORDER BY p.id,s.scene,s.id',parameters)
        with tracking(rows,'Read retained portrait identities',unit='assignment',position=0) as assignments:
            for row in assignments:
                target=ready.setdefault(row['id'],dict(person=row['id'],performer=row['performer'],scans=[]))
                item=dict(scene=row['scene'],scan=row['scan'])
                target['scans'].append(item)
    for target in ready.values():
        target['scans'].sort(key=lambda item:(item['scene'],item['scan']))
    ready=dict(sorted(ready.items()))
    return ready
