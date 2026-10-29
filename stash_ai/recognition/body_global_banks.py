"""Metadata-only global banks. Embeddings are read lazily by candidate checks."""
import json

from .body_identity import independent,policy
from .body_observations import contract
from .database import encoded
from .identity_merges import canonical
from .memory_size import size
from .scan_heads import predicate as current_scan

ROUTES=('body_global','body_scene')


def banks(db,config,scope,namespace):
    expected=contract(config);settings=config.data['sapiensid']['identities']
    limit=config.data['sapiensid']['execution']['reference_bytes'];used=0
    def reserve(extra):
        if used+extra>limit:
            raise ValueError('Global body reference metadata exceeds sapiensid.execution.reference_bytes.')
    where=("s.scope=? AND s.namespace=? AND s.status='complete' AND b.contract=? AND b.rejected IS NULL "
           "AND m.status='accepted' AND m.role='anchor' AND m.policy=? AND "+current_scan())
    args=(scope,namespace,expected,policy(config))
    source=' FROM body_observations b JOIN body_members m ON m.observation=b.id JOIN scans s ON s.id=b.scan '
    owners={}
    for row in db.execute('SELECT DISTINCT m.person'+source+'WHERE '+where,args):
        owners.setdefault(canonical(db,row[0]),[]).append(row[0])
    used=size(owners);reserve(0);output={}
    # Fetch quality for one identity at a time, never its crop/embedding blobs.
    for person,aliases in owners.items():
        rows=[];working=0
        for raw in db.execute('SELECT b.id,b.scan,b.cue,b.quality,length(b.embedding) AS bytes'+source+
            'WHERE '+where+' AND m.person IN (SELECT value FROM json_each(?)) ORDER BY b.scan,b.cue,b.id',
            (*args,encoded(aliases))):
            if raw['bytes']!=4096*4:raise ValueError('Invalid retained person embedding.')
            row=dict(raw);working+=size(row);reserve(working);rows.append(row)
        witnesses=independent(rows,settings)
        selected=witnesses[:config.data['sapiensid']['global_matching']['max_references']]
        if not selected and rows:selected=rows[:1]
        refs=[dict(id=r['id'],scan=r['scan'],cue=r['cue']) for r in selected]
        output[person]=dict(refs=refs,cores=[refs],missing_original=False,support=len(witnesses),
                            faces=[],face_ids=[],cues=set())
        used+=size(output[person])+size(person)+128;reserve(0)

    def body(identifier):
        raw=db.execute('SELECT b.id,b.scan,b.cue,length(b.embedding) AS bytes FROM body_observations b '
            'JOIN scans s ON s.id=b.scan WHERE b.id=? AND b.contract=? AND b.rejected IS NULL AND '+current_scan(),
            (identifier,expected)).fetchone()
        if raw is None or raw['bytes'] is None:return None
        if raw['bytes']!=4096*4:raise ValueError('Invalid retained person embedding.')
        return dict(id=raw['id'],scan=raw['scan'],cue=raw['cue'])

    # Preserve proof multiplicity for future replays, while identical cores
    # share metadata storage and the exact-score cache.
    signatures={p:{tuple(r['id'] for r in b['refs']):b['refs']} for p,b in output.items()}
    original_faces={p:set() for p in output}
    for alias in db.execute('SELECT target,evidence FROM identity_merges'):
        proof=json.loads(alias['evidence']);target=canonical(db,alias['target'])
        if proof.get('route') not in ROUTES or target not in output:continue
        bank=output[target]
        for ids in proof.get('cores',{}).values():
            signature=tuple(ids)
            if signature in signatures[target]:
                if signatures[target][signature]:bank['cores'].append(signatures[target][signature])
                used+=16;reserve(0);continue
            core=[]
            for identifier in ids:
                row=body(identifier)
                if row is None:bank['missing_original']=True;continue
                used+=size(row)+16;reserve(0);core.append(row)
            if core:bank['cores'].append(core)
            signatures[target][signature]=core
        for ids in proof.get('face_witnesses',{}).values():original_faces[target].update(ids)

    from .reference_roles import strict_sql
    from .pose_policy import anchor_sql
    for person,bank in output.items():
        face_ids=set(original_faces[person])
        face_ids.update(r[0] for r in db.execute('SELECT o.id FROM observations o JOIN members m ON m.observation=o.id '
            'JOIN scans s ON s.id=o.scan WHERE m.person=? AND o.rejected IS NULL AND '+strict_sql()+' AND '+
            anchor_sql('o',config.data['clustering'])+' AND '+current_scan(),(person,)))
        # Unassigned but measured faces linked to body cores retain veto power.
        face_ids.update(r[0] for r in db.execute('SELECT b.face FROM body_observations b '
            'JOIN body_members m ON m.observation=b.id JOIN scans s ON s.id=b.scan '
            'JOIN observations o ON o.id=b.face WHERE m.person=? AND b.rejected IS NULL '
            'AND o.rejected IS NULL AND '+current_scan(),(person,)))
        for identifier in sorted(face_ids):
            raw=db.execute('SELECT o.id,o.dimensions FROM observations o JOIN scans s ON s.id=o.scan '
                'WHERE o.id=? AND o.rejected IS NULL AND '+current_scan(),(identifier,)).fetchone()
            if raw is None:bank['missing_original']=True;continue
            row=dict(raw);used+=size(row)+64;reserve(0)
            bank['faces'].append(row);bank['face_ids'].append(identifier)
        cues=bank['cues']
        def cue(scan,index):
            nonlocal used
            value=(scan,index)
            if value not in cues:
                used+=size(value)+64;reserve(0);cues.add(value)
        for core in bank['cores']:
            for row in core:cue(row['scan'],row['cue'])
        for row in db.execute('SELECT b.scan,b.cue FROM body_observations b JOIN body_members m ON m.observation=b.id '
            'JOIN scans s ON s.id=b.scan WHERE m.person=? AND m.status=\'accepted\' AND '+current_scan(),(person,)):
            cue(row['scan'],row['cue'])
        for row in db.execute('SELECT o.scan,o.cue FROM observations o JOIN members m ON m.observation=o.id '
            'JOIN scans s ON s.id=o.scan WHERE m.person=? AND o.rejected IS NULL AND '+current_scan(),(person,)):
            cue(row['scan'],row['cue'])
    return output
