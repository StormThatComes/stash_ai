"""Transactional scene and owner clocks, including old addresses on REPLACE."""

def bump(kind,query):
    return (f"INSERT INTO evidence_versions SELECT '{kind}',address,1 FROM ({query}) WHERE address IS NOT NULL "
            "GROUP BY address ON CONFLICT(kind,key) DO UPDATE SET version=version+1;")


def triggers():
    result={}
    for table in ('scans','observations','members','candidates','people','person_evidence',
                  'body_observations','body_cues','body_members','body_origins','identity_origins','identity_merges'):
        for event in ('INSERT','UPDATE','DELETE'):
            sides=('new',) if event=='INSERT' else ('old',) if event=='DELETE' else ('old','new')
            body=''
            for side in sides:
                scans=[];people=[]
                if table=='scans':
                    # Inserting a new head can remove old owners from admission.
                    scans=[f'SELECT id FROM scans WHERE scope={side}.scope AND namespace={side}.namespace AND scene={side}.scene']
                elif table in ('observations','candidates','body_observations','body_cues'):
                    scans=[f'SELECT {side}.scan AS id']
                else:
                    if table in ('members','person_evidence','identity_origins'):
                        scans=[f'SELECT scan AS id FROM observations WHERE id={side}.observation']
                    elif table in ('body_members','body_origins'):
                        scans=[f'SELECT scan AS id FROM body_observations WHERE id={side}.observation']
                if table in ('people','identity_merges'):
                    people=[side+'.id'] if table=='people' else [side+'.source',side+'.target']
                    scans=[f'SELECT o.scan AS id FROM observations o JOIN members m ON m.observation=o.id WHERE m.person IN ({",".join(people)})',
                           f'SELECT b.scan AS id FROM body_observations b JOIN body_members m ON m.observation=b.id WHERE m.person IN ({",".join(people)})',
                           f'SELECT b.scan AS id FROM body_observations b JOIN body_origins i ON i.observation=b.id WHERE i.person IN ({",".join(people)})']
                if table=='candidates':
                    scans.append(f"SELECT scan AS id FROM observations WHERE id=CASE WHEN json_valid({side}.payload) THEN json_extract({side}.payload,'$.observation') END")
                if table in ('members','body_members','body_origins','identity_origins','candidates'):
                    people.append(side+'.person')
                for scan in scans:
                    body+=bump('scene',f'SELECT json_array(s.scope,s.namespace,s.scene) AS address FROM scans s WHERE s.id IN ({scan})')
                    # Owner clocks cover all admitted face/body inputs in the scene;
                    # bounded scene fan-out also catches strict-role/head changes.
                    body+=bump('owner',f'SELECT m.person AS address FROM members m JOIN observations o ON o.id=m.observation WHERE o.scan IN ({scan}) '
                               f'UNION SELECT m.person FROM body_members m JOIN body_observations b ON b.id=m.observation WHERE b.scan IN ({scan})')
                for owner in people:body+=bump('owner',f'SELECT {owner} AS address')
                if table=='identity_merges':body+=bump('alias-head',"SELECT '*' AS address")
            operation=event
            if event=='UPDATE' and table=='people':operation='UPDATE OF id,scope,namespace,created,token'
            if event=='UPDATE' and table=='observations':operation='UPDATE OF id,scan,cue,seconds,geometry,quality,rejected,dimensions,embedding'
            name=f'recognition_local_{table}_{event.lower()}'
            result[name]=f'CREATE TRIGGER {name} AFTER {operation} ON {table} BEGIN '+body+' END'
        # REPLACE deletes old addresses without DELETE triggers by default.
        key='observation' if table in ('members','person_evidence','body_members','body_origins','identity_origins') else 'source' if table=='identity_merges' else 'id'
        if table in ('body_cues','identity_origins'):continue
        old=result[f'recognition_local_{table}_delete']
        body=old.split(' BEGIN ',1)[1].rsplit(' END',1)[0]
        # Before insert, re-expose the conflicting row as old via scalar queries.
        import re
        body=re.sub(r'old\.([a-z_]+)',lambda m:f'(SELECT {m[1]} FROM {table} WHERE {key}=new.{key} LIMIT 1)',body)
        name=f'recognition_local_{table}_replace'
        result[name]=f'CREATE TRIGGER {name} BEFORE INSERT ON {table} BEGIN '+body+' END'
    for event in ('INSERT','UPDATE','DELETE'):
        sides=('new',) if event=='INSERT' else ('old',) if event=='DELETE' else ('old','new')
        name='recognition_local_head_'+event.lower()
        condition=' OR '.join(side+".key LIKE 'recognition-scan-head:%'" for side in sides)
        result[name]=f'CREATE TRIGGER {name} AFTER {event} ON metadata WHEN '+condition+' BEGIN '+bump('heads',"SELECT '*' AS address")+' END'
    return result


TRIGGERS=triggers()
SCHEMA=';\n'.join(TRIGGERS.values())+';\n'


def scene_key(db,scan):
    row=db.execute('SELECT json_array(scope,namespace,scene) FROM scans WHERE id=?',(scan,)).fetchone()
    return row[0] if row else scan


def available(db):
    return db.execute("SELECT 1 FROM sqlite_master WHERE name='recognition_local_members_insert'").fetchone() is not None
