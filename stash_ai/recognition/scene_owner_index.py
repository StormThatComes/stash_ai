"""Transactional counts of distinct scene owners, including external SQLite writers.

The small owner table replaces repeated observation/member joins in invalidation
triggers. A keyed ledger makes REPLACE correct with recursive_triggers on or off.
"""

TABLES = ('scene_owner_members', 'scene_owners')
TABLE_SCHEMA = '''
CREATE TABLE scene_owner_members(kind TEXT, observation TEXT, scan TEXT, person TEXT,
 PRIMARY KEY(kind,observation));
CREATE INDEX scene_owner_members_scan ON scene_owner_members(scan);
CREATE TABLE scene_owners(scan TEXT,person TEXT,total INTEGER NOT NULL,
 PRIMARY KEY(scan,person),CHECK(total>0));
'''


def triggers():
    result = {}
    def add(name, event, table, body, when=''):
        result[name] = f'CREATE TRIGGER {name} AFTER {event} ON {table} {when} BEGIN {body} END'
    remove = ("DELETE FROM scene_owners WHERE scan=old.scan AND person=old.person AND total=1;"
              "UPDATE scene_owners SET total=total-1 WHERE scan=old.scan AND person=old.person;")
    insert = ("INSERT INTO scene_owners VALUES(new.scan,new.person,1) "
              "ON CONFLICT(scan,person) DO UPDATE SET total=total+1;")
    for event, body in (('INSERT', insert), ('DELETE', remove), ('UPDATE', remove+insert)):
        add('recognition_owner_ledger_'+event.lower(), event, 'scene_owner_members', body,
            'WHEN old.scan IS NOT new.scan OR old.person IS NOT new.person' if event=='UPDATE' else '')
    for kind, members, observations in (('face','members','observations'),
                                         ('body','body_members','body_observations')):
        put = (f"INSERT INTO scene_owner_members SELECT '{kind}',new.observation,o.scan,new.person "
               f"FROM {observations} o WHERE o.id=new.observation "
               "ON CONFLICT(kind,observation) DO UPDATE SET scan=excluded.scan,person=excluded.person;")
        delete = f"DELETE FROM scene_owner_members WHERE kind='{kind}' AND observation=old.observation;"
        for event, body in (('INSERT',put),('DELETE',delete),('UPDATE',delete+put)):
            add(f'recognition_owner_{kind}_{event.lower()}',event,members,body)
        for event in ('INSERT','UPDATE OF id,scan'):
            add(f'recognition_owner_{kind}_address_'+event.split()[0].lower(),event,observations,
                f"UPDATE scene_owner_members SET scan=new.scan WHERE kind='{kind}' AND observation=new.id;")
    return result


TRIGGERS = triggers()
SCHEMA = TABLE_SCHEMA+';\n'.join(TRIGGERS.values())+';\n'


def populate(db):
    """Called with ledger triggers absent, during migration/private bulk import."""
    db.execute('DELETE FROM scene_owner_members')
    db.execute('DELETE FROM scene_owners')
    for kind, members, observations in (('face','members','observations'),('body','body_members','body_observations')):
        db.execute(f"INSERT INTO scene_owner_members SELECT '{kind}',m.observation,o.scan,m.person "
                   f'FROM {members} m JOIN {observations} o ON o.id=m.observation')
    db.execute('INSERT INTO scene_owners SELECT scan,person,COUNT(*) FROM scene_owner_members GROUP BY scan,person')
