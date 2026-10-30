"""Metadata-only global banks. Embeddings are read lazily by candidate checks."""
from .resource_errors import ResourceLimit
import json

from .body_identity import independent,policy
from .body_observations import contract,compatible_contracts
from .database import encoded
from .identity_merges import canonical
from .memory_size import size
from .scan_heads import predicate as current_scan

ROUTES=('body_global','body_scene')


def banks(db,config,scope,namespace):
    from .run_cache import current, database_key, MISSING
    from .config import digest, plain
    from .body_cache import matching_settings
    cache=current();generation=database_key(db)
    policy_key=digest(matching_settings(config))
    key=('body-global-banks',generation,scope,namespace,policy_key)
    found=cache.get(key) if generation is not None else MISSING
    if found is not MISSING:return found
    found=_banks(db,config,scope,namespace)
    if generation is not None:
        cache.put(('body-global-banks',database_key(db),scope,namespace,policy_key),found,size(found))
    return found


def _banks(db,config,scope,namespace):
    expected=contract(config);settings=config.data['sapiensid']['identities']
    limit=config.data['sapiensid']['execution']['reference_bytes'];used=0
    def reserve(extra):
        if used+extra>limit:
            raise ResourceLimit('Global body reference metadata exceeds sapiensid.execution.reference_bytes.')
    where=("s.scope=? AND s.namespace=? AND s.status='complete' AND b.contract IN (SELECT value FROM json_each(?)) AND b.rejected IS NULL "
           "AND m.status IN ('accepted','provisional') AND m.role='anchor' AND "+current_scan())
    args=(scope,namespace,encoded(compatible_contracts(config)))
    source=' FROM body_observations b JOIN body_members m ON m.observation=b.id JOIN scans s ON s.id=b.scan '
    owners={}
    for row in db.execute('SELECT DISTINCT m.person'+source+'WHERE '+where,args):
        owners.setdefault(canonical(db,row[0]),[]).append(row[0])
    used=size(owners);reserve(0);output={}
    # Bounded owner batches eliminate a metadata query for every identity.
    def owner_rows():
        people=list(owners)
        for start in range(0,len(people),32):
            batch=people[start:start+32]
            aliases={alias:person for person in batch for alias in owners[person]}
            groups={person:[] for person in batch};working=0
            for raw in db.execute('SELECT m.person AS owner,b.id,b.scan,b.cue,b.quality,m.policy,m.status,length(b.embedding) AS bytes'+source+
                'WHERE '+where+' AND m.person IN (SELECT value FROM json_each(?)) ORDER BY b.scan,b.cue,b.id',
                (*args,encoded(list(aliases)))):
                if raw['bytes']!=4096*4:raise ValueError('Invalid retained person embedding.')
                row=dict(raw);person=aliases[row.pop('owner')]
                working+=size(row);reserve(working);groups[person].append(row)
            for person in batch:
                rows=groups.pop(person)
                yield person,rows
                working-=sum(size(r) for r in rows)
                reserve(working)
    for person,rows in owner_rows():
        witnesses=independent(rows,settings)
        selected=witnesses[:config.data['sapiensid']['global_matching']['max_references']]
        if not selected and rows:selected=rows[:1]
        # Deduplication controls support, never an outlier's veto authority.
        core=[dict(id=r['id'],scan=r['scan'],cue=r['cue']) for r in rows]
        by_id={r['id']:r for r in core}
        refs=[by_id[r['id']] for r in selected]
        output[person]=dict(refs=refs,cores=[core],missing_original=False,support=len(witnesses),
                            faces=[],face_ids=[],cues=set(),witnesses=witnesses,provisional=all(r['status']=='provisional' for r in rows),stale_policy=any(r['policy']!=policy(config) for r in rows))
        used+=size(output[person])+size(person)+128;reserve(0)

    def bodies(identifiers):
        yield from metadata_rows(db,identifiers,body=True,contracts=compatible_contracts(config),reserve=reserve)

    # Preserve proof multiplicity for future replays, while identical cores
    # share metadata storage and the exact-score cache.
    signatures={p:{tuple(r['id'] for r in b['cores'][0]):b['cores'][0]} for p,b in output.items()}
    original_faces={p:set() for p in output}
    confirmation_witnesses={p:set() for p in output}
    for alias in db.execute('SELECT target,evidence FROM identity_merges'):
        proof=json.loads(alias['evidence']);target=canonical(db,alias['target'])
        if proof.get('route') not in ROUTES or target not in output:continue
        bank=output[target]
        if proof.get('provisional_confirmation'):
            confirmation_witnesses[target].update(i for ids in proof.get('confirmation_witnesses',proof.get('cores',{})).values() for i in ids)
        for owner,ids in proof.get('cores',{}).items():
            if proof.get('body_witness_format',0)<2:
                # Earlier proofs retained a capped or independence-filtered core.
                # Original anchor roles remain in attachment decision payloads.
                extra=[r[0] for r in db.execute('SELECT b.id FROM body_observations b '
                    'JOIN body_members m ON m.observation=b.id WHERE b.contract IN (SELECT value FROM json_each(?)) '
                    "AND json_extract(m.proof,'$.global_merge.source')=? "
                    "AND json_extract(m.proof,'$.role')='anchor'",(encoded(compatible_contracts(config)),owner))]
                ids=list(dict.fromkeys([*ids,*extra]))
            signature=tuple(ids)
            if signature in signatures[target]:
                if signatures[target][signature]:bank['cores'].append(signatures[target][signature])
                used+=16;reserve(0);continue
            core=[]
            for row in bodies(ids):
                if row is None:bank['missing_original']=True;continue
                used+=size(row)+16;reserve(0);core.append(row)
            if core:bank['cores'].append(core)
            signatures[target][signature]=core
        for ids in proof.get('face_witnesses',{}).values():original_faces[target].update(ids)

    from .reference_roles import strict_sql
    from .pose_policy import anchor_sql
    selected_owners=encoded(list(output))
    for raw in db.execute('SELECT m.person,o.id FROM observations o JOIN members m ON m.observation=o.id '
        'JOIN scans s ON s.id=o.scan WHERE m.person IN (SELECT value FROM json_each(?)) '
        'AND o.rejected IS NULL AND '+strict_sql()+' AND '+
        anchor_sql('o',config.data['clustering'])+' AND '+current_scan(),(selected_owners,)):
        original_faces[raw['person']].add(raw['id'])
    # Linked unassigned faces retain their veto authority as before.
    for raw in db.execute('SELECT m.person,b.face FROM body_observations b '
        'JOIN body_members m ON m.observation=b.id JOIN scans s ON s.id=b.scan '
        'JOIN observations o ON o.id=b.face WHERE m.person IN (SELECT value FROM json_each(?)) '
        'AND b.rejected IS NULL AND o.rejected IS NULL AND '+current_scan(),(selected_owners,)):
        original_faces[raw['person']].add(raw['face'])
    used+=size(original_faces);reserve(0)
    def cue(person,scan,index):
        nonlocal used
        cues=output[person]['cues'];value=(scan,index)
        if value not in cues:
            used+=size(value)+64;reserve(0);cues.add(value)
    for person,bank in output.items():
        face_ids=original_faces[person]
        for raw in metadata_rows(db,sorted(face_ids),body=False,reserve=reserve):
            if raw is None:bank['missing_original']=True;continue
            row=dict(raw);used+=size(row)+64;reserve(0)
            bank['faces'].append(row);bank['face_ids'].append(row['id'])
        # Original cores still supply independent support after consolidation.
        extra=[]
        for row in bodies(sorted(confirmation_witnesses[person])):
            if row is None:bank['missing_original']=True
            else:extra.append(row);used+=size(row)+16;reserve(0)
        bank['witnesses']=independent(bank['witnesses']+extra,settings)
        bank['support']=len(bank['witnesses'])
        for core in bank['cores']:
            for row in core:cue(person,row['scan'],row['cue'])
    for row in db.execute('SELECT m.person,b.scan,b.cue FROM body_observations b JOIN body_members m ON m.observation=b.id '
        'JOIN scans s ON s.id=b.scan WHERE m.person IN (SELECT value FROM json_each(?)) '
        "AND m.status='accepted' AND "+current_scan(),(selected_owners,)):
        cue(row['person'],row['scan'],row['cue'])
    for row in db.execute('SELECT m.person,o.scan,o.cue FROM observations o JOIN members m ON m.observation=o.id '
        'JOIN scans s ON s.id=o.scan WHERE m.person IN (SELECT value FROM json_each(?)) '
        'AND o.rejected IS NULL AND '+current_scan(),(selected_owners,)):
        cue(row['person'],row['scan'],row['cue'])
    return output


def metadata_rows(db, identifiers, *, body, contracts=(), reserve=lambda _:None):
    """Ordered bounded keyed reads; no embedding or crop crosses into Python."""
    from itertools import islice
    iterator=iter(identifiers)
    while batch:=list(islice(iterator,256)):
        if body:
            query=('SELECT b.id,b.scan,b.cue,b.quality,length(b.embedding) AS bytes FROM body_observations b '
                'JOIN scans s ON s.id=b.scan WHERE b.id IN (SELECT value FROM json_each(?)) '
                'AND b.contract IN (SELECT value FROM json_each(?)) AND b.rejected IS NULL AND '+current_scan())
            args=(encoded(batch),encoded(contracts))
        else:
            query=('SELECT o.id,o.dimensions FROM observations o JOIN scans s ON s.id=o.scan '
                'WHERE o.id IN (SELECT value FROM json_each(?)) AND o.rejected IS NULL AND '+current_scan())
            args=(encoded(batch),)
        rows={};working=size(batch)
        reserve(working)
        for raw in db.execute(query,args):
            row=dict(raw)
            if body:
                amount=row.pop('bytes')
                if amount is None:continue
                if amount!=4096*4:raise ValueError('Invalid retained person embedding.')
            working+=size(row)+128;reserve(working);rows[row['id']]=row
        for identifier in batch:
            yield rows.get(identifier)
            # The caller may retain the yielded metadata in its own bank.
            reserve(working)
