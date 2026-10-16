"""Private replay-owned canonical membership order and representative snapshots.

Only the cold-replay coordinator owns these triggers and mutable state. Any
transaction failure aborts its scope; nothing is reused after rollback.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from heapq import merge

from .database import encoded
from .profiling import count, measured

_active = ContextVar('recognition_replay_references', default=None)


class ReferenceIndex:
    def __init__(self, db, scope, namespace, config):
        self.db, self.scope, self.namespace = db, scope, namespace
        self.members, self.deltas, self.groups = {}, {}, {}
        self.dirty = set()
        self.roots = None
        self.cached = None
        total = db.execute('SELECT COUNT(*) FROM observations').fetchone()[0]
        # Conservative allowance for IDs, Python dictionaries, ranks and owners;
        # embeddings remain owned by SQLite and the bounded representatives.
        self.enabled = total * 512 <= config.data['execution']['rebuild_memory_bytes']
        if not self.enabled:
            count('reference_index_budget_fallbacks')
            return
        rows = db.execute('SELECT o.id,s.scope,s.namespace,s.status,o.rejected FROM observations o '
            'JOIN scans s ON s.id=o.scan ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,'
            'o.seconds,o.geometry,o.embedding,o.id')
        self.order, self.rank, self.allowed = [], {}, set()
        from .progress import tracking
        with tracking(rows, 'Index canonical replay references', unit='observation',
                      total=total, position=0) as tracked:
            for row in tracked:
                self.rank[row['id']] = len(self.order)
                self.order.append(row['id'])
                if (row['scope'] == scope and row['namespace'] == namespace
                        and row['status'] == 'complete' and row['rejected'] is None):
                    self.allowed.add(row['id'])
        for row in db.execute('SELECT observation,person FROM members'):
            self.changed(row['observation'], None, row['person'])
        db.create_function('recognition_reference_member', 3, self.changed)
        db.create_function('recognition_reference_roots', 0, self.changed_roots)
        db.executescript('''
          CREATE TEMP TRIGGER recognition_reference_insert AFTER INSERT ON members BEGIN
            SELECT recognition_reference_member(new.observation,NULL,new.person); END;
          CREATE TEMP TRIGGER recognition_reference_delete AFTER DELETE ON members BEGIN
            SELECT recognition_reference_member(old.observation,old.person,NULL); END;
          CREATE TEMP TRIGGER recognition_reference_update AFTER UPDATE ON members BEGIN
            SELECT recognition_reference_member(old.observation,old.person,NULL);
            SELECT recognition_reference_member(new.observation,NULL,new.person); END;
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
        for observation, (old, new) in self.deltas.items():
            if old == new:
                continue
            if observation not in self.allowed:
                raise ValueError('Face identity has incompatible retained membership evidence.')
            rank = self.rank[observation]
            if old is not None:
                changes.setdefault(old, (set(), set()))[0].add(rank)
            if new is not None:
                changes.setdefault(new, (set(), set()))[1].add(rank)
        self.deltas.clear()
        for person, (removed, added) in changes.items():
            retained = self.members.get(person, [])
            # New scenes usually append in canonical order. Merge/retraction
            # batches preserve exact order without sorting their full history.
            if not removed and (not retained or not added or retained[-1] < min(added)):
                retained.extend(sorted(added))
                self.members[person] = retained
            else:
                old = (rank for rank in retained if rank not in removed and rank not in added)
                self.members[person] = list(merge(old, sorted(added)))
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
            raise ValueError('Face identity lookup exceeds configured person budget.')
        identity = tuple(people), policy['max_samples_per_person']
        if self.cached is not None and self.cached[0] == identity:
            changed = any(tuple(self.order[rank] for rank in self.members.get(person, [])[
                                :policy['max_samples_per_person']]) != self.groups.get(person, ((),))[0]
                          for person in self.dirty)
            if not changed:
                _, groups, size, samples = self.cached
                usage.add_counts(size, samples, people=len(groups), kind='Retained')
                self.dirty.clear()
                count('reference_snapshot_reuses')
                return list(groups)
        self.cached = None
        wanted = {}
        refreshed = 0
        for person in people:
            ids = tuple(self.order[rank] for rank in self.members[person][:policy['max_samples_per_person']])
            previous = self.groups.get(person)
            if previous is None or previous[0] != ids:
                refreshed += 1
            wanted[person] = ids
        desired = {identifier for ids in wanted.values() for identifier in ids}
        # Reuse unchanged vectors and release obsolete representatives before
        # allocating replacements. Never hold two complete reference sets.
        values = {identifier: value for ids, samples in self.groups.values()
                  for identifier, value in zip(ids, samples) if identifier in desired}
        self.groups.clear()
        usage.add(list(values.values()), kind='Retained')
        # One bounded vector fetch per group batch; never fetch full history.
        identifiers = list(dict.fromkeys(identifier for ids in wanted.values() for identifier in ids
                                        if identifier not in values))
        for offset in range(0, len(identifiers), 256):
            for row in self.db.execute('SELECT id,dimensions,embedding FROM observations WHERE id IN '
                    '(SELECT value FROM json_each(?))', (encoded(identifiers[offset:offset+256]),)):
                value = vector(row)
                usage.add([value], kind='Retained')
                values[row['id']] = value
        for person, ids in wanted.items():
            self.groups[person] = ids, [values[identifier] for identifier in ids]
        self.groups = {person: self.groups[person] for person in people}
        self.dirty.clear()
        result = []
        for person in people:
            samples = self.groups[person][1]
            usage.add([], people=1, kind='Retained')
            result.append(Group(person, tuple(samples), True))
        count('reference_snapshot_reuses')
        count('reference_people_refreshed', refreshed)
        self.cached = identity, tuple(result), usage.bytes, usage.samples
        return result

    def close(self):
        if not self.enabled:
            return
        for name in ('insert', 'delete', 'update', 'people', 'alias'):
            self.db.execute('DROP TRIGGER IF EXISTS temp.recognition_reference_' + name)
        self.db.create_function('recognition_reference_member', 3, None)
        self.db.create_function('recognition_reference_roots', 0, None)


def current(db):
    index = _active.get()
    return index if index is not None and index.enabled and index.db is db else None


@contextmanager
def references(db, scope, namespace, config):
    from stash_ai.runtime.cleanup import cleanup
    index = ReferenceIndex(db, scope, namespace, config)
    token = _active.set(index)
    try:
        yield index
    finally:
        _active.reset(token)
        cleanup([('close replay reference index', index.close)])
