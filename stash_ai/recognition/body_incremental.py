"""Transactional reuse of body centroids; live rivals are always recomputed."""
import numpy as np
from .config import digest
from .dependencies import supported,version
from .derived_cache import current
from .dependency_schema import validate_triggers
from .resource_errors import ResourceLimit


def source_versions(db):
    if not supported(db) or db.execute('PRAGMA user_version').fetchone()[0]<9:return None
    validate_triggers(db,required=True)
    return dict(db.execute("SELECT key,version FROM evidence_versions WHERE kind='raw-body'"))


def means_key(bank,config,versions):
    if versions is None:return None
    rows=list(bank['refs'])+[r for core in bank['cores'] for r in core]
    return digest(['body-means-v1',bank['refs'],bank['cores'],bool(bank['faces']),
        [r['id'] for r in bank['witnesses']],config.data['sapiensid']['global_matching'],
        sorted({(r['scan'],versions.get(r['scan'],0)) for r in rows})])


def nominations(search):
    """Compare changed owners to every live owner; reuse certified old negatives."""
    store=current()
    settings=search.config.data['sapiensid']['global_matching']
    floor=settings['match_threshold']
    key=digest(['body-centroids-v1',floor])
    versions=search.mean_versions
    previous=store.get('body-centroid-catalogue',key) if store else None
    old=previous['owners'] if previous else {}
    changed={p for p,v in versions.items() if old.get(p)!=v}
    if previous is None or len(changed)*4>max(1,len(versions)):
        # Preserve efficient bounded BLAS tiles and geometric exclusion for a
        # cold/mostly changed catalogue. Delta rows are for small updates.
        pairs=list(search.dense_nominations({}))
        if store:store.put('body-centroid-catalogue',key,dict(owners=versions,pairs=pairs))
        yield from pairs
        return
    pairs={tuple(p) for p in previous['pairs'] if all(x in versions and x not in changed for x in p)} if previous else set()
    limit=search.config.data['sapiensid']['execution']['max_candidate_pairs']
    budget=search.config.data['sapiensid']['execution']['plan_bytes']
    if len(pairs)>limit or len(pairs)*256>budget:
        raise ResourceLimit('Global body nominations exceed configured candidate or plan budget.')
    def admit(pair):
        pairs.add(pair)
        if len(pairs)>limit or len(pairs)*256>budget:
            raise ResourceLimit('Global body nominations exceed configured candidate or plan budget.')
    # Read bounded tiles, with only a single changed row on the left. This also
    # avoids evaluating the unchanged portion of the same centroid tile.
    from .body_global_search import ROUNDING
    for index,(owner,_) in enumerate(search.entries):
        person=search.identities[owner]
        if person not in changed:continue
        search.index.seek(index*4096*8)
        left=np.frombuffer(search.index.read(4096*8),dtype='<f8')
        for start in range(0,len(search.entries),search.block):
            owners=search.mode_owners[start:start+search.block]
            chosen=[j for j,o in enumerate(owners) if o!=owner and
                    (search.identities[o] not in changed or index<start+j)]
            if not chosen:continue
            search.index.seek(start*4096*8)
            right=np.frombuffer(search.index.read(len(owners)*4096*8),dtype='<f8').reshape(-1,4096)[chosen]
            scores=left@right.T
            search.stats['centroid_pairs']+=len(chosen)
            for j,score in zip(chosen,scores):
                if score+ROUNDING>=floor:admit(tuple(sorted((person,search.identities[owners[j]]))))
    if previous and len(changed)<len(versions):search.stats['centroid_catalogue_hits']=len(versions)-len(changed)
    if store:store.put('body-centroid-catalogue',key,dict(owners=versions,pairs=sorted(pairs)))
    for pair in sorted(pairs):
        search.stats['candidate_pairs']+=1
        yield pair
