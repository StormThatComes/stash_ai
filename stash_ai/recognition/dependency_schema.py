"""Compiled, audited cache-invalidation DDL; never execute database-supplied SQL."""

TABLE_SCHEMA = '''
CREATE TABLE evidence_versions(kind TEXT NOT NULL, key TEXT NOT NULL,
 version INTEGER NOT NULL CHECK(typeof(version)='integer' AND version>0),
 PRIMARY KEY(kind,key));
CREATE TABLE evidence_fingerprints(kind TEXT NOT NULL, key TEXT NOT NULL, policy TEXT NOT NULL,
 version INTEGER NOT NULL CHECK(typeof(version)='integer' AND version>=0),
 digest TEXT NOT NULL, PRIMARY KEY(kind,key,policy));
DROP INDEX IF EXISTS candidates_reference_observation;
CREATE INDEX candidates_reference_observation ON candidates(
 CAST(CASE WHEN json_valid(payload) THEN json_extract(payload,'$.observation') END AS TEXT));
'''


def _bump(kind, key):
    return (f"INSERT INTO evidence_versions(kind,key,version) SELECT '{kind}',{key},1 WHERE {key} IS NOT NULL "
            "ON CONFLICT(kind,key) DO UPDATE SET version=version+1;")


def triggers():
    output={}
    for table in ('scans','observations','candidates','members','people','identity_merges',
                  'identity_origins','merge_support','models','metadata'):
        for event in ('INSERT','UPDATE','DELETE'):
            sides=('new',) if event=='INSERT' else ('old',) if event=='DELETE' else ('old','new')
            body=[_bump('global',"'*'")]
            if table in ('people','identity_merges','identity_origins'):
                body.append(_bump('bindings',"'*'"))
            for side in sides:
                if table=='scans':
                    body.append(_bump('scan',side+'.id'))
                elif table in ('observations','candidates'):
                    body.append(_bump('scan',side+'.scan'))
                if table in ('members','candidates'):
                    address=(side+'.observation' if table=='members' else
                             f"CASE WHEN json_valid({side}.payload) THEN json_extract({side}.payload,'$.observation') END")
                    body.append(_bump('scan',f'(SELECT scan FROM observations WHERE id={address})'))
            when=(" WHEN "+' OR '.join(s+".key='installation'" for s in sides)) if table=='metadata' else ''
            # Portrait JPEG updates do not alter identity evidence.
            operation=("UPDATE OF id,scan,cue,seconds,source,rect,geometry,quality,rejected,dimensions,embedding"
                       if event=='UPDATE' and table=='observations' else event)
            name='recognition_version_'+table+'_'+event.lower()
            output[name]=f'CREATE TRIGGER {name} AFTER {operation} ON {table}{when} BEGIN '+''.join(body)+' END'
    # SQLite REPLACE can delete the conflicting row without DELETE triggers
    # when recursive_triggers is off. Capture its old addresses before insert,
    # including writes made by an independent SQLite connection.
    replacements={
        'observations':[_bump('scan','(SELECT scan FROM observations WHERE id=new.id)')],
        'candidates':[_bump('scan','(SELECT scan FROM candidates WHERE id=new.id)'),
            _bump('scan',"(SELECT o.scan FROM candidates c JOIN observations o ON o.id="
                  "CASE WHEN json_valid(c.payload) THEN json_extract(c.payload,'$.observation') END WHERE c.id=new.id)")],
        'scans':[_bump('scan','(SELECT id FROM scans WHERE id=new.id)'),
                 _bump('scan','(SELECT id FROM scans WHERE scope=new.scope AND namespace=new.namespace '
                       'AND scene=new.scene AND input_sha=new.input_sha AND policy_sha=new.policy_sha)')],
    }
    for table,body in replacements.items():
        name='recognition_version_'+table+'_replace'
        output[name]=f'CREATE TRIGGER {name} BEFORE INSERT ON {table} BEGIN '+''.join(body)+' END'
    return output


TRIGGERS=triggers()
SCHEMA=TABLE_SCHEMA+'\n'+';\n'.join(TRIGGERS.values())+';\n'


EXTRA_TRIGGERS={}
for table in ('person_evidence','evidence_thumbnails'):
    for event in ('INSERT','UPDATE','DELETE'):
        sides=('new',) if event=='INSERT' else ('old',) if event=='DELETE' else ('old','new')
        body=[_bump('global',"'*'")]
        if table=='person_evidence':
            for side in sides:body.append(_bump('scan',f'(SELECT scan FROM observations WHERE id={side}.observation)'))
        name='recognition_version_'+table+'_'+event.lower()
        EXTRA_TRIGGERS[name]=f'CREATE TRIGGER {name} AFTER {event} ON {table} BEGIN '+''.join(body)+' END'
EXTRA_SCHEMA=';\n'.join(EXTRA_TRIGGERS.values())+';\n'

def validate_triggers(db, *, required=False):
    """Reject unknown/mutated executable schema, including spoofed names."""
    accepted={**TRIGGERS,**EXTRA_TRIGGERS}
    rows=list(db.execute("SELECT type,name,sql FROM sqlite_master WHERE type IN ('trigger','view')"))
    normalize=lambda value:' '.join(value.strip().rstrip(';').split())
    for kind,name,sql in rows:
        if kind!='trigger' or name not in accepted or normalize(sql)!=normalize(accepted[name]):
            raise ValueError('Unsupported facial recognition schema triggers or views.')
    expected=set(TRIGGERS)
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='evidence_thumbnails'").fetchone():expected.update(EXTRA_TRIGGERS)
    if required and not expected<={row[1] for row in rows}:
        raise ValueError('Incomplete facial recognition dependency triggers; restore a consistent backup.')
