"""Private replay-owned canonical membership order and representative snapshots.

Only the cold-replay coordinator owns these triggers and mutable state. Any
transaction failure aborts its scope; nothing is reused after rollback.
"""
from .resource_errors import ResourceLimit
from contextlib import contextmanager
from contextvars import ContextVar
from heapq import merge
from array import array

from .database import encoded
from .profiling import count, measured

_active = ContextVar('recognition_replay_references', default=None)


class ReferenceIndex:
    def __init__(self, db, scope, namespace, config):
        self.db, self.scope, self.namespace = db, scope, namespace
        self.members, self.deltas, self.groups = {}, {}, {}
        self.sampler=None
        self.enabled=False
        self.role_dirty, self.owners = set(), {}
        self.incremental, self.scene_by_id = {}, {}
        from .reference_roles import index
        index(db)
        from .pose_policy import is_anchor
        self.is_anchor = lambda q: is_anchor(q, config.data['clustering'])
        self.dirty = set()
        self.roots = None
        self.cached = None
        total = db.execute('SELECT COUNT(*) FROM observations').fetchone()[0]
        # Conservative allowance for IDs, Python dictionaries, ranks and owners;
        # embeddings remain owned by SQLite and the bounded representatives.
        from .run_cache import current
        cache=current()
        available=(cache.working-sum(cache.reservations.values()) if cache is not None and cache.working is not None
                   else config.data['execution']['rebuild_memory_bytes'])
        owners = db.execute('SELECT COUNT(*) FROM people').fetchone()[0]
        scenes = db.execute('SELECT COUNT(*) FROM scans').fetchone()[0]
        extra = db.execute('SELECT COALESCE(SUM(MAX(0,length(id)-36)*5),0) FROM observations').fetchone()[0]
        required = total*224 + 256*(owners+scenes) + extra + 524288
        self.enabled = required <= available
        if self.enabled and cache is not None:
            cache.reserve(('reference-index',id(self)), required)
        if not self.enabled:
            count('reference_index_budget_fallbacks')
            return
        import sqlite3
        from .scratch_space import connect
        self.sampler=connect()
        self.sampler.execute('PRAGMA cache_size=-256')
        self.sampler.execute('CREATE TABLE state(owner TEXT PRIMARY KEY,maximum INTEGER,payload BLOB)')
        rows = db.execute('SELECT o.id,o.quality,s.scope,s.namespace,s.scene,s.status,o.rejected FROM observations o '
            'JOIN scans s ON s.id=o.scan ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,'
            'o.seconds,o.geometry,o.embedding,o.id')
        from .packed_ids import Order,Ranks,Flags,Values
        self.order,self.rank=Order(),Ranks()
        self.allowed,self.anchors=Flags(self.rank,total),Flags(self.rank,total)
        self.scene_by_id,self.owners=Values(self.rank,total),Values(self.rank,total)
        from .progress import tracking
        with tracking(rows, 'Index canonical replay references', unit='observation',
                      total=total, position=0) as tracked:
            for row in tracked:
                self.rank[row['id']] = len(self.order)
                self.order.append(row['id'])
                self.scene_by_id[row['id']] = row['scene']
                if (row['scope'] == scope and row['namespace'] == namespace
                        and row['status'] == 'complete' and row['rejected'] is None):
                    self.allowed.add(row['id'])
                    if self.is_anchor(row['quality']):
                        self.anchors.add(row['id'])
        from .reference_roles import strict_sql
        for row in db.execute('SELECT m.observation,m.person,'+strict_sql()+
                ' AS strict FROM members m JOIN observations o ON o.id=m.observation'):
            identifier=row['observation']
            if identifier not in self.allowed:raise ValueError('Face identity has incompatible retained membership evidence.')
            if identifier in self.anchors and row['strict']:
                self.members.setdefault(row['person'],array('I')).append(self.rank[identifier])
                self.owners[identifier]=row['person'];self.dirty.add(row['person'])
        for person,ranks in self.members.items():self.members[person]=array('I',sorted(ranks))
        db.create_function('recognition_reference_member', 3, self.changed)
        db.create_function('recognition_reference_roots', 0, self.changed_roots)
        db.create_function('recognition_reference_role', 1, self.role_dirty.add)
        db.executescript('''
          CREATE TEMP TRIGGER recognition_reference_insert AFTER INSERT ON members BEGIN
            SELECT recognition_reference_member(new.observation,NULL,new.person); END;
          CREATE TEMP TRIGGER recognition_reference_delete AFTER DELETE ON members BEGIN
            SELECT recognition_reference_member(old.observation,old.person,NULL); END;
          CREATE TEMP TRIGGER recognition_reference_update AFTER UPDATE ON members BEGIN
            SELECT recognition_reference_member(old.observation,old.person,NULL);
            SELECT recognition_reference_member(new.observation,NULL,new.person); END;
          CREATE TEMP TRIGGER recognition_reference_candidate_insert AFTER INSERT ON candidates BEGIN
            SELECT recognition_reference_role(CASE WHEN json_valid(new.payload) THEN json_extract(new.payload,'$.observation') END); END;
          CREATE TEMP TRIGGER recognition_reference_candidate_update AFTER UPDATE ON candidates BEGIN
            SELECT recognition_reference_role(CASE WHEN json_valid(new.payload) THEN json_extract(new.payload,'$.observation') END); END;
          CREATE TEMP TRIGGER recognition_reference_candidate_delete AFTER DELETE ON candidates BEGIN
            SELECT recognition_reference_role(CASE WHEN json_valid(old.payload) THEN json_extract(old.payload,'$.observation') END); END;
          CREATE TEMP TRIGGER recognition_reference_people AFTER INSERT ON people BEGIN
            SELECT recognition_reference_roots(); END;
          CREATE TEMP TRIGGER recognition_reference_alias AFTER INSERT ON identity_merges BEGIN
            SELECT recognition_reference_roots(); END;
        ''')

    def changed_roots(self):
        self.roots = None

    def changed(self, observation, old, new):
        previous = self.deltas.get(observation)
        self.deltas[observation] = (previous[0] if previous is not None else old, new)

    def flush(self):
        changes = {}
        from .reference_roles import strict_sql
        affected = sorted((set(self.deltas) | self.role_dirty) - {None})
        actual = {}
        for offset in range(0, len(affected), 256):
            for row in self.db.execute('SELECT o.id,m.person,' + strict_sql() + ' AS strict '
                    'FROM observations o JOIN members m ON m.observation=o.id '
                    'WHERE o.id IN (SELECT value FROM json_each(?))',
                    (encoded(affected[offset:offset+256]),)):
                if row['id'] not in self.allowed:
                    raise ValueError('Face identity has incompatible retained membership evidence.')
                if row['id'] in self.anchors and row['strict']:
                    actual[row['id']] = row['person']
        for observation in affected:
            old, new = self.owners.get(observation), actual.get(observation)
            if old == new:
                continue
            rank = self.rank[observation]
            if old is not None:
                changes.setdefault(old, (set(), set()))[0].add(rank)
            if new is not None:
                changes.setdefault(new, (set(), set()))[1].add(rank)
                self.owners[observation] = new
            else:
                self.owners.pop(observation, None)
        self.role_dirty.clear()
        self.deltas.clear()
        for person, (removed, added) in changes.items():
            retained = self.members.get(person, array('I'))
            append_only = (not removed and retained and added and retained[-1] < min(added)
                and self.scene_by_id[self.order[retained[-1]]] != self.scene_by_id[self.order[min(added)]])
            if append_only and self.incremental.get(person, []) is not None:
                self.incremental.setdefault(person, []).extend(sorted(added))
            else:
                self.incremental[person] = None
            # New scenes usually append in canonical order. Merge/retraction
            # batches preserve exact order without sorting their full history.
            if not removed and (not retained or not added or retained[-1] < min(added)):
                retained.extend(sorted(added))
                self.members[person] = retained
            else:
                old = (rank for rank in retained if rank not in removed and rank not in added)
                self.members[person] = array('I',merge(old, sorted(added)))
            self.dirty.add(person)

    @measured('reference_index_refresh')
    def snapshot(self, policy, usage):
        from .clustering import Group, vector
        self.flush()
        if self.roots is None:
            self.roots = [row['id'] for row in self.db.execute(
                'SELECT id FROM people WHERE scope=? AND namespace=? AND NOT EXISTS '
                '(SELECT 1 FROM identity_merges WHERE source=people.id) ORDER BY created,id',
                (self.scope, self.namespace))]
        people = [person for person in self.roots if self.members.get(person)]
        if len(people) > policy['max_people_per_lookup']:
            raise ResourceLimit('Face identity lookup exceeds configured person budget.')
        identity = tuple(people), policy['max_samples_per_person']
        if self.cached is not None and self.cached[0] == identity:
            if not self.dirty:
                _, groups, size, samples = self.cached
                from .paged_references import Samples
                usage.add_counts(size, samples, people=len(groups), kind='Retained',
                                 paged=all(isinstance(g.samples,Samples) for g in groups))
                self.dirty.clear()
                count('reference_snapshot_reuses')
                return list(groups)
        same_cap = self.cached is not None and self.cached[0][1] == policy['max_samples_per_person']
        self.cached = None
        wanted = {}
        refreshed = 0
        for person in people:
            previous = self.groups.get(person)
            if previous is not None and person not in self.dirty and same_cap:
                ids = previous[0]
            else:
                from .reference_selection import select
                from .pose_policy import anchor_sql
                from .reference_roles import strict_sql
                # Full canonical refresh of changed owners: previous bounded
                # representatives are not a sufficient summary for resampling.
                import pickle
                added=self.incremental.get(person)
                saved=self.sampler.execute('SELECT maximum,payload FROM state WHERE owner=?',(person,)).fetchone()
                state=pickle.loads(saved[1]) if added and saved and saved[0]==policy['max_samples_per_person'] else {}
                rows = self.canonical_rows(person,ranks=added if state else None)
                ids = tuple(identifier for identifier, _ in select(rows, policy['max_samples_per_person'],state=state))
                self.sampler.execute('INSERT OR REPLACE INTO state VALUES (?,?,?)',
                    (person,policy['max_samples_per_person'],pickle.dumps(state,protocol=5)))
            previous = self.groups.get(person)
            if previous is None or previous[0] != ids:
                refreshed += 1
            wanted[person] = ids
        from .paged_references import pack
        previous_groups=self.groups
        self.groups={}
        for person,ids in wanted.items():
            previous=previous_groups.pop(person,None)
            if previous is not None and previous[0]==ids:
                samples=previous[1]
            else:
                values={}
                for offset in range(0,len(ids),256):
                    for row in self.db.execute('SELECT id,dimensions,embedding FROM observations WHERE id IN '
                        '(SELECT value FROM json_each(?))',(encoded(ids[offset:offset+256]),)):
                        values[row['id']]=vector(row)
                samples=pack(values[i] for i in ids)
            usage.add(samples,kind='Retained')
            self.groups[person]=(ids,samples)
        self.dirty.clear()
        self.incremental.clear()
        result = []
        for person in people:
            samples = self.groups[person][1]
            usage.add([], people=1, kind='Retained')
            result.append(Group(person, samples, True, self.groups[person][0]))
        count('reference_snapshot_reuses')
        count('reference_people_refreshed', refreshed)
        self.cached = identity, tuple(result), usage.bytes, usage.samples
        return result

    def canonical_rows(self, person, *, ranks=None):
        """Use indexed membership ranks and the shared immutable row cache.

        Recompute selection from every member, without an SQL sort or correlated
        membership predicate. Cache eviction only changes the number of reads.
        """
        from .run_cache import current, MISSING
        cache = current()
        ranks = self.members[person] if ranks is None else ranks
        for offset in range(0, len(ranks), 256):
            ids = [self.order[rank] for rank in ranks[offset:offset+256]]
            values, missing = {}, []
            for identifier in ids:
                key = ('reference-row', id(self), identifier)
                found = cache.get(key) if cache is not None else MISSING
                if found is MISSING:
                    missing.append(identifier)
                else:
                    values[identifier] = found
            if missing:
                for row in self.db.execute('SELECT o.id,o.scan,o.dimensions,o.embedding,o.quality,o.geometry,o.seconds,s.scene '
                        'FROM observations o JOIN scans s ON s.id=o.scan WHERE o.id IN '
                        '(SELECT value FROM json_each(?))', (encoded(missing),)):
                    value = dict(row)
                    values[row['id']] = value
                    if cache is not None:
                        cache.put(('reference-row', id(self), row['id']), value,
                                  len(row['embedding']) + len(row['quality']) + len(row['geometry']) + 1024)
            yield from (values[identifier] for identifier in ids)

    def close(self):
        if self.sampler is not None:self.sampler.close();self.sampler=None
        if not self.enabled:
            return
        for name in ('insert', 'delete', 'update', 'people', 'alias',
                     'candidate_insert', 'candidate_update', 'candidate_delete'):
            self.db.execute('DROP TRIGGER IF EXISTS temp.recognition_reference_' + name)
        from .run_cache import current
        cache = current()
        if cache is not None:
            cache.discard(lambda key: key[0] == 'reference-row' and key[1] == id(self))
            cache.reserve(('reference-index',id(self)),0)
        self.db.create_function('recognition_reference_member', 3, None)
        self.db.create_function('recognition_reference_roots', 0, None)
        self.db.create_function('recognition_reference_role', 1, None)


class LazyIndex:
    def __init__(self,*args):self.args=args;self.db=args[0];self.value=None
    def get(self):
        if self.value is None:
            value=ReferenceIndex.__new__(ReferenceIndex)
            try:value.__init__(*self.args)
            except BaseException:
                value.close();raise
            self.value=value
        return self.value
    def close(self):
        if self.value is not None:self.value.close()


def current(db):
    index = _active.get()
    if index is None or index.db is not db:return None
    if isinstance(index,LazyIndex):index=index.get()
    return index if index.enabled else None


@contextmanager
def references(db, scope, namespace, config):
    from stash_ai.runtime.cleanup import cleanup
    index = LazyIndex(db, scope, namespace, config)
    token = _active.set(index)
    try:
        yield index
    finally:
        _active.reset(token)
        cleanup([('close replay reference index', index.close)])
