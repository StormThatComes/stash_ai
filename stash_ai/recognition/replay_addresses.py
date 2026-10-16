"""Derived candidate addresses owned only by a trusted private cold replay."""
from contextlib import contextmanager
from contextvars import ContextVar

from stash_ai.runtime.cleanup import cleanup

_active = ContextVar('recognition_candidate_addresses', default=None)


def address(db, alias=''):
    prefix = alias + '.' if alias else ''
    if _active.get() is db:
        return f'(SELECT observation FROM candidate_addresses WHERE candidate={prefix}id)'
    return f"json_extract({prefix}payload,'$.observation')"


@contextmanager
def addresses(db):
    db.executescript('''
      CREATE TEMP TABLE candidate_addresses(candidate TEXT PRIMARY KEY, observation TEXT NOT NULL);
      INSERT INTO candidate_addresses SELECT id,json_extract(payload,'$.observation') FROM candidates;
      CREATE TEMP TRIGGER candidate_address_insert AFTER INSERT ON candidates BEGIN
        INSERT INTO candidate_addresses VALUES (new.id,json_extract(new.payload,'$.observation')); END;
      CREATE TEMP TRIGGER candidate_address_delete AFTER DELETE ON candidates BEGIN
        DELETE FROM candidate_addresses WHERE candidate=old.id; END;
      CREATE TEMP TRIGGER candidate_address_update AFTER UPDATE OF payload,id ON candidates
        WHEN old.id IS NOT new.id OR json_extract(old.payload,'$.observation') IS NOT json_extract(new.payload,'$.observation')
        BEGIN DELETE FROM candidate_addresses WHERE candidate=old.id;
        INSERT INTO candidate_addresses VALUES (new.id,json_extract(new.payload,'$.observation')); END;
    ''')
    token = _active.set(db)
    try:
        yield
    finally:
        _active.reset(token)
        cleanup([(f'drop candidate address {suffix}',
                  lambda suffix=suffix: db.execute('DROP TRIGGER IF EXISTS temp.candidate_address_' + suffix))
                 for suffix in ('insert','delete','update')] +
                [('drop candidate addresses',lambda: db.execute('DROP TABLE temp.candidate_addresses'))])
