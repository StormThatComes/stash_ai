"""Select existing portrait targets without recognition or Stash publication."""
from collections.abc import Mapping

from stash_ai.runtime.console import activity
from .database import encoded
from .models import namespace
from .progress import tracking


class LazyAssets(Mapping):
    """Cached portrait reranking needs no model installation or device setup."""
    def __init__(self, config, factory, report):
        self.config, self.factory, self.report = config, factory, report
        self.paths = None

    def _load(self):
        if self.paths is None:
            with activity(self.report, 'prepare portrait model assets'):
                self.paths = self.factory(self.config, write=self.report.write)
        return self.paths

    def __getitem__(self, key):
        return self._load()[key]

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())


def retained_targets(database, config, scope, scene_ids=None):
    """Current memberships only; never resurrect historical candidate decisions.

    The normal canonical-target pass additionally excludes roots with unresolved
    performer merges. Model namespaces and Stash instances cannot be mixed.
    """
    rows = database.connection.execute('''SELECT DISTINCT p.id,p.performer,s.scene,s.id AS scan
      FROM people p JOIN members m ON m.person=p.id
      JOIN observations o ON o.id=m.observation JOIN scans s ON s.id=o.scan
      WHERE p.scope=? AND p.namespace=? AND p.performer IS NOT NULL
        AND s.scope=p.scope AND s.namespace=p.namespace AND s.status='complete'
        AND o.rejected IS NULL
        AND NOT EXISTS (SELECT 1 FROM identity_merges a WHERE a.source=p.id)
        AND (? IS NULL OR s.scene IN (SELECT value FROM json_each(?)))
      ORDER BY p.id,s.scene,s.id''',
      (scope, namespace(config), None if scene_ids is None else encoded(scene_ids),
       None if scene_ids is None else encoded(scene_ids)))
    ready = {}
    with tracking(rows, 'Read retained portrait identities', unit='assignment', position=0) as assignments:
        for row in assignments:
            target = ready.setdefault(row['id'], {'person': row['id'], 'performer': row['performer'], 'scans': []})
            target['scans'].append({'scene': row['scene'], 'scan': row['scan']})
    return ready
