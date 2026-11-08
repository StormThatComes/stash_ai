"""Certified owner/perspective selections reused across disk gallery generations."""
import hashlib
import json
from .database import encoded
from .dependencies import version
from .profiling import count


def selections(source, pages, records, tracks, witness, settings, scores, rank, local_shot, cache, policy):
    signature = hashlib.sha256(policy.encode())
    scans = []
    for raw in records:
        row = witness(raw)
        row['roots'] = json.loads(row['roots'])
        if not scans or scans[-1]!=row['scan']:
            scans.append(row['scan'])
            if cache is not None:
                signature.update(encoded([row['scan'],version(source,'raw-body',row['scan'])]).encode())
        if cache is not None:
            signature.update(encoded(row).encode())
        tracks.add(row,local_shot(row))
    key = signature.hexdigest()
    kind = 'body-appearance-rear-manifest'
    manifest = cache.get(kind,key) if cache is not None else None
    if manifest is not None:
        count('rear_gallery_reused_groups')
        for source_scan,ids in manifest:
            rows = []
            for oid in ids:
                stored = pages.execute('SELECT payload FROM witnesses WHERE id=?',(oid,)).fetchone()
                rows.append(witness(json.loads(stored[0])))
            yield source_scan,rows
        return
    count('rear_gallery_rebuilt_groups')
    manifest = [] if cache is not None else None
    charge = 0
    from itertools import chain
    for scan in chain(scans,(None,) if len(scans)>1 else ()):
        for kept in tracks.modes(scan,settings,scores,rank,local_shot if scan else lambda row:row['scan']):
            if len(kept)<max(2,settings['attach_min_support']):
                continue
            rows = [r for _,_,r in kept]
            label = rows[0]['scan']
            if manifest is not None:
                ids = [r['id'] for r in rows]
                charge += sum(len(i)+48 for i in ids)+128
                if charge>1024*1024:
                    manifest = None
                else:
                    manifest.append((label,ids))
            yield label,rows
    if cache is not None and manifest is not None:
        cache.put(kind,key,manifest)
