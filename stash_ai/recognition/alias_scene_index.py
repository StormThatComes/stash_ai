"""Indexed historical scene dependencies and revalidation requests for body aliases."""
import json

TABLES = ('body_alias_scenes','pending_body_merges')
STATEMENTS = (
    'CREATE TABLE body_alias_scenes(source TEXT,scene TEXT,PRIMARY KEY(source,scene))',
    'CREATE TABLE pending_body_merges(scope TEXT,namespace TEXT,people TEXT,policy TEXT,scenes TEXT,'
    'versions TEXT,PRIMARY KEY(scope,namespace,people))',
)
def _insert(source):
    return (f"INSERT OR IGNORE INTO body_alias_scenes SELECT {source}.source,CAST(j.value AS TEXT) "
            f"FROM json_each({source}.evidence,'$.scenes') j WHERE "
            f"json_extract({source}.evidence,'$.route') IN ('body_global','body_scene');")

TRIGGERS = {
    'recognition_alias_scene_insert': 'CREATE TRIGGER recognition_alias_scene_insert AFTER INSERT ON identity_merges '
    'BEGIN DELETE FROM body_alias_scenes WHERE source=new.source;'+_insert('new')+' END',
    'recognition_alias_scene_update': 'CREATE TRIGGER recognition_alias_scene_update AFTER UPDATE ON identity_merges '
    'BEGIN DELETE FROM body_alias_scenes WHERE source IN (old.source,new.source);'+_insert('new')+' END',
    'recognition_alias_scene_delete': 'CREATE TRIGGER recognition_alias_scene_delete AFTER DELETE ON identity_merges '
    'BEGIN DELETE FROM body_alias_scenes WHERE source=old.source; END',
}
SCHEMA = ';\n'.join((*STATEMENTS,*TRIGGERS.values()))+';\n'


def populate(db):
    db.execute('DELETE FROM body_alias_scenes')
    db.execute("INSERT OR IGNORE INTO body_alias_scenes SELECT m.source,CAST(j.value AS TEXT) "
               "FROM identity_merges m,json_each(m.evidence,'$.scenes') j "
               "WHERE json_extract(m.evidence,'$.route') IN ('body_global','body_scene')")


def inherited(db, people):
    # Only canonical destinations qualify. Walk backwards through the existing
    # target index, including intermediate aliases of any route.
    ancestors = """WITH RECURSIVE owners(id) AS (
        SELECT value FROM json_each(?) WHERE NOT EXISTS (SELECT 1 FROM identity_merges WHERE source=value)
        UNION SELECT m.source FROM owners o JOIN identity_merges m ON m.target=o.id
    ) """
    if db.execute('PRAGMA user_version').fetchone()[0]>=17:
        sql = 'SELECT scene FROM owners o JOIN body_alias_scenes s ON s.source=o.id'
    else:
        sql = ("SELECT j.value FROM owners o JOIN identity_merges m ON m.source=o.id,json_each(m.evidence,'$.scenes') j "
               "WHERE json_extract(m.evidence,'$.route') IN ('body_global','body_scene')")
    return {r[0] for r in db.execute(ancestors+sql,(people,))}


def remember(db,scope,namespace,policy,people,scenes):
    from .database import encoded
    from .dependencies import version
    owners = sorted(people)
    clocks = {owner:version(db,'owner',owner) for owner in owners}
    db.execute('INSERT OR REPLACE INTO pending_body_merges VALUES (?,?,?,?,?,?)',
               (scope,namespace,encoded(owners),policy,encoded(sorted(scenes,key=int)),encoded(clocks)))


def expire(db,scope,namespace,policy,selected):
    from .database import encoded
    # A full selection consumes a nomination even if the cold plan disproves it.
    db.execute("""DELETE FROM pending_body_merges WHERE scope=? AND namespace=? AND
        (policy<>? OR NOT EXISTS (SELECT 1 FROM json_each(scenes) WHERE value NOT IN
            (SELECT value FROM json_each(?))) OR EXISTS (SELECT 1 FROM json_each(people) j
            WHERE NOT EXISTS (SELECT 1 FROM people p WHERE p.id=j.value)))""",
        (scope,namespace,policy,encoded(sorted(selected))))


def pending(db,scope,namespace,scene,limit):
    if db.execute('PRAGMA user_version').fetchone()[0]<17:
        return []
    return [dict(people=json.loads(p),required_scenes=json.loads(s),policy=policy,owner_versions=json.loads(v))
            for p,s,policy,v in db.execute('SELECT people,scenes,policy,versions FROM pending_body_merges '
                'WHERE scope=? AND namespace=? AND ? IN (SELECT value FROM json_each(scenes)) '
                'ORDER BY people LIMIT ?', (scope,namespace,scene,limit))]
