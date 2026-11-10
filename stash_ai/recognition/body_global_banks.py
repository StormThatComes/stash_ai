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


def diverse_references(witnesses, maximum, reserve=lambda _: None):
    """Bounded scan/appearance coverage, then evenly spaced independent witnesses."""
    import hashlib
    import heapq
    def address(row):
        return {key: row[key] for key in ('id','scan','cue')}
    if len(witnesses) <= maximum:
        reserve(768 * len(witnesses))
        return [address(row) for row in witnesses]
    # A rare rear/side view must survive a flood of frontal appearances. Each
    # orientation holds a bounded deterministic sample of scan/appearance strata.
    buckets = {}
    retained = 0
    for row in witnesses:
        quality = json.loads(row['quality']).get('quality', {})
        direction = quality.get('orientation', 'unknown')
        if direction not in ('front', 'side', 'back'):
            direction = 'unknown'
        representatives, heap = buckets.setdefault(direction, ({}, []))
        bucket = (row['scan'], str(quality.get('appearance_hash') or '')[:2])
        rank = int.from_bytes(hashlib.blake2b(encoded(bucket).encode(), digest_size=8).digest(), 'big')
        if rank in representatives or len(heap) == maximum and rank >= -heap[0]:
            continue
        if len(heap) == maximum:
            del representatives[-heapq.heappop(heap)]
        else:
            retained += 1
            reserve(768 * retained)
        representatives[rank] = address(row)
        heapq.heappush(heap, -rank)
    queues = []
    for direction in ('front','side','back','unknown'):
        if direction in buckets:
            records, _ = buckets[direction]
            queues.append(iter([records[k] for k in sorted(records)]))
    chosen = {}
    while queues and len(chosen) < maximum:
        remaining = []
        for queue in queues:
            row = next(queue, None)
            if row is not None:
                chosen[row['id']] = row
                remaining.append(queue)
            if len(chosen) == maximum:
                break
        queues = remaining
    step = max(1, len(witnesses) // maximum)
    for offset in range(0, len(witnesses), step):
        if len(chosen) == maximum:
            break
        row = witnesses[offset]
        if row['id'] not in chosen:
            retained += 1
            reserve(768 * retained)
            chosen[row['id']] = address(row)
    return sorted(chosen.values(), key=lambda row: (row['scan'], row['cue'], row['id']))


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
    from .paged_references import rows as page_rows,cues as page_cues
    from .config import digest
    from .reference_roles import strict_sql
    from .pose_policy import anchor_sql
    expected=contract(config);settings=config.data['sapiensid']['identities']
    limit=config.data['sapiensid']['execution']['reference_bytes']
    where=("s.scope=? AND s.namespace=? AND s.status='complete' AND b.contract IN (SELECT value FROM json_each(?)) AND b.rejected IS NULL "
           "AND m.status IN ('accepted','provisional') AND m.role='anchor' AND "+current_scan())
    args=(scope,namespace,encoded(compatible_contracts(config)))
    source=' FROM body_observations b JOIN body_members m ON m.observation=b.id JOIN scans s ON s.id=b.scan '
    owners={}
    for row in db.execute('SELECT DISTINCT m.person'+source+'WHERE '+where,args):
        owners.setdefault(canonical(db,row[0]),[]).append(row[0])
    output={};retained=size(owners)
    def reserve(extra):
        if retained+extra>limit:raise ResourceLimit('Global body reference metadata exceeds sapiensid.execution.reference_bytes.')
    # Resolve merge targets once; per-owner proof reads use the target index.
    proofs={}
    for row in db.execute('SELECT source,target FROM identity_merges'):
        target=canonical(db,row['target'])
        if target in owners:proofs.setdefault(target,[]).append(row['source'])
    for person,aliases in owners.items():
        def source_rows():
            for raw in db.execute('SELECT b.id,b.scan,b.cue,b.quality,m.policy,m.status,length(b.embedding) AS bytes'+source+
                'WHERE '+where+' AND m.person IN (SELECT value FROM json_each(?)) ORDER BY b.scan,b.cue,b.id',(*args,encoded(aliases))):
                if raw['bytes']!=16384:raise ValueError('Invalid retained person embedding.')
                yield dict(raw)
        rows=page_rows(source_rows())
        witnesses=independent(rows,settings,presorted=True,paged=True,reserve=reserve)
        selected=diverse_references(witnesses,config.data['sapiensid']['global_matching']['max_references'],reserve)
        if not selected and rows:selected=rows[:1]
        core=page_rows(dict(id=r['id'],scan=r['scan'],cue=r['cue']) for r in rows)
        selected_ids={r['id'] for r in selected}
        references={r['id']:r for r in core if r['id'] in selected_ids}
        bank=dict(refs=[references[r['id']] for r in selected],cores=[core],missing_original=False,
            support=len(witnesses),faces=[],face_ids=[],cues=page_cues(),witnesses=witnesses,
            provisional=all(r['status']=='provisional' for r in rows),stale_policy=any(r['policy']!=policy(config) for r in rows))
        # Hash the ordered identifiers incrementally, matching digest(list(ids)).
        import hashlib
        signature=hashlib.sha256();signature.update(b'[')
        for index,row in enumerate(core):
            if index:signature.update(b',')
            signature.update(encoded(row['id']).encode())
        signature.update(b']')
        signatures={signature.hexdigest():core}
        reserve(size(bank));rows=core=witnesses=None
        face_ids=set();confirmation=set()
        for merge in db.execute('SELECT evidence FROM identity_merges WHERE source IN (SELECT value FROM json_each(?))',
                                (encoded(proofs.get(person,[])),)):
            proof=json.loads(merge[0])
            if proof.get('route') not in ROUTES:continue
            # Early endpoint merges persisted lists of cores rather than flat addresses.
            cores = {owner:list(dict.fromkeys(oid for value in ids
                     for oid in (value if isinstance(value,list) else [value])))
                     for owner,ids in proof.get('cores',{}).items()}
            if proof.get('provisional_confirmation'):
                confirmation.update(i for ids in proof.get('confirmation_witnesses',cores).values() for i in ids)
            for owner,ids in cores.items():
                if proof.get('body_witness_format',0)<2:
                    extra=[r[0] for r in db.execute('SELECT b.id FROM body_observations b JOIN body_members m ON m.observation=b.id '
                        'WHERE b.contract IN (SELECT value FROM json_each(?)) AND json_extract(m.proof,\'$.global_merge.source\')=? '
                        "AND json_extract(m.proof,'$.role')='anchor'",(encoded(compatible_contracts(config)),owner))]
                    ids=list(dict.fromkeys([*ids,*extra]))
                signature=digest(ids)
                if signature in signatures:
                    if signatures[signature]:bank['cores'].append(signatures[signature])
                    continue
                core=[]
                for row in metadata_rows(db,ids,body=True,contracts=compatible_contracts(config),reserve=reserve):
                    if row is None:bank['missing_original']=True
                    else:core.append(row)
                signatures[signature]=page_rows(core)
                if core:bank['cores'].append(signatures[signature])
            for ids in proof.get('face_witnesses',{}).values():face_ids.update(ids)
        face_ids.update(r[0] for r in db.execute('SELECT o.id FROM observations o JOIN members m ON m.observation=o.id '
            'JOIN scans s ON s.id=o.scan WHERE m.person=? AND o.rejected IS NULL AND '+strict_sql()+' AND '+
            anchor_sql('o',config.data['clustering'])+' AND '+current_scan(),(person,)))
        face_ids.update(r[0] for r in db.execute('SELECT b.face FROM body_observations b JOIN body_members m ON m.observation=b.id '
            'JOIN scans s ON s.id=b.scan JOIN observations o ON o.id=b.face WHERE m.person=? '
            'AND b.rejected IS NULL AND o.rejected IS NULL AND '+current_scan(),(person,)))
        reserve(size(face_ids))
        for raw in metadata_rows(db,sorted(face_ids),body=False,reserve=reserve):
            if raw is None:bank['missing_original']=True
            else:bank['faces'].append(dict(raw));bank['face_ids'].append(raw['id'])
        bank['faces']=page_rows(bank['faces']);bank['face_ids']=page_rows(bank['face_ids'])
        extra=[]
        for row in metadata_rows(db,sorted(confirmation),body=True,contracts=compatible_contracts(config),reserve=reserve):
            if row is None:bank['missing_original']=True
            else:extra.append(row)
        if extra:bank['witnesses']=page_rows(independent(bank['witnesses']+extra,settings))
        bank['support']=len(bank['witnesses'])
        for core in bank['cores']:
            for row in core:bank['cues'].add((row['scan'],row['cue']))
        for table,address,condition in (('body_observations','body_members',"m.status='accepted'"),('observations','members','b.rejected IS NULL')):
            for scan,cue in db.execute(f'SELECT b.scan,b.cue FROM {table} b JOIN {address} m ON m.observation=b.id '
                'JOIN scans s ON s.id=b.scan WHERE m.person=? AND '+condition+' AND '+current_scan(),(person,)):
                bank['cues'].add((scan,cue))
        retained+=size(bank)+size(person)+128;reserve(0);output[person]=bank
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
