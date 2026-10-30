"""One publication's frozen certificate, with cheap audited dependency checks."""
from copy import deepcopy
from .database import encoded
from .dependencies import supported, version, snapshot as read_snapshot
from .dependency_schema import validate_triggers


class Snapshot:
    def __init__(self, db, scope, namespace, scene, config=None):
        self.db,self.scope,self.namespace,self.scene,self.config=db,scope,namespace,scene,config
        self.initial_key=self.value=self.schema=None

    def _key(self):
        from .publication_queue import latest, matching_policy, body_policy, current
        db=self.db
        if not supported(db):return None
        schema=db.execute('PRAGMA schema_version').fetchone()[0]
        if schema!=self.schema:
            validate_triggers(db,required=True)
            self.schema=schema
        scan=latest(db,self.scope,self.namespace,self.scene)
        if scan is None:return None
        identifier=scan['id']
        # Profile performer IDs change during publication; identity ownership
        # and tokens do not. Read those small fields instead of every vector.
        ids={r[0] for r in db.execute('SELECT m.person FROM members m JOIN observations o '
                                     'ON o.id=m.observation WHERE o.scan=?',(identifier,))}
        body_ids=()
        from .body_identity import available
        if available(db):
            ids.update(r[0] for r in db.execute('SELECT m.person FROM body_members m JOIN body_observations b '
                                                'ON b.id=m.observation WHERE b.scan=?',(identifier,)))
            # Old-schema REPLACE triggers do not stamp the previous scene when
            # a body ID moves. Its indexed address set closes that gap.
            body_ids=tuple(r[0] for r in db.execute('SELECT id FROM body_observations WHERE scan=? ORDER BY id',(identifier,)))
        bindings=tuple(tuple(r) for r in db.execute('SELECT id,scope,namespace,token FROM people '
            'WHERE id IN (SELECT value FROM json_each(?)) ORDER BY id',(encoded(sorted(ids-{None})),)))
        queued=current(db,self.scope,self.namespace,self.scene) or {}
        policy=matching_policy(self.config) if self.config is not None else queued.get('policy')
        body=body_policy(self.config) if self.config is not None else queued.get('body')
        return (schema,db.execute('PRAGMA data_version').fetchone()[0],identifier,
                version(db,'scan',identifier),bindings,body_ids,policy,encoded(body))

    def read(self):
        from .publication_queue import snapshot
        from .profiling import count
        with read_snapshot(self.db):
            key=self._key()
            if key is not None and self.value is not None and key==self.initial_key:
                count('publication_snapshot_hits')
                return deepcopy(self.value)
            count('publication_snapshot_reads')
            value=snapshot(self.db,self.scope,self.namespace,self.scene,self.config)
            if self.value is None and key is not None:
                self.initial_key,self.value=key,deepcopy(value)
            # Never advance the original certificate: rolled-back transactions
            # may reuse a generation number for different subsequent writes.
            return value
