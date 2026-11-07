"""Attach faceless rear queries to certified external views in one generation."""
from collections import Counter, defaultdict
import json
import numpy as np

from .body_identity import policy, quality, vector
from .database import encoded
from .scan_heads import predicate


def _global_rival(search, query, ignored, floor):
    # The disk index contains exact means of every production body mode,
    # including provisional appearances. Round borderline results toward veto.
    for start in range(0,len(search.entries),search.block):
        count = min(search.block,len(search.entries)-start)
        search.index.seek(start*4096*8)
        block = np.frombuffer(search.index.read(count*4096*8),dtype='<f8').reshape(count,4096)
        scores = block @ query
        for offset in np.flatnonzero(scores+1e-6>=floor):
            owner = search.identities[search.entries[start+int(offset)][0]]
            if owner not in ignored:
                return owner
    return None


def _rows(db, scan, stamp):
    return db.execute("""SELECT b.id,b.cue,b.scan,b.face,b.quality,b.geometry,b.embedding,
        m.person,m.status,m.role,m.proof FROM body_observations b JOIN body_members m
        ON m.observation=b.id WHERE b.scan=? AND m.policy=? AND b.rejected IS NULL
        ORDER BY b.cue,b.id""",(scan,stamp))


def _scene(db, scan, stamp, gallery, search, config):
    from .body_perspectives import orientation
    settings = config.data['sapiensid']['identities']
    proposals = {}
    required = defaultdict(set)
    satellites = defaultdict(dict)
    face_owners = set()
    for raw in _rows(db,scan,stamp):
        row = dict(raw)
        old = row['person']
        decision = json.loads(row['proof'])
        if old is not None:
            if row['face']:
                face_owners.add(old)
            if row['role']=='anchor' or decision.get('reason')=='body_scene_core':
                required[old].add(row['id'])
            else:
                satellites[old][row['id']] = dict(decision,person=old,status=row['status'],role=row['role'])
        if (row['face'] or row['status']=='accepted' and decision.get('reason') not in ('body_scene_core','body_core_attachment')
                or decision.get('track_blocker',{}).get('reason') in ('competing_transition','different_face_root')):
            continue
        person = json.loads(row['geometry'])
        kind = orientation(person) if person.get('keypoints') else quality(row).get('orientation')
        if kind != 'back':
            continue
        query = vector(row)
        ranked = gallery.rank(query,scan,settings,quality(row).get('pixel_digest'))
        if not ranked:
            continue
        mean,low,owner,source,witnesses,_ = ranked[0]
        if (owner==old or decision.get('open_face_root',{}).get('owner',owner)!=owner
                or mean<settings['match_threshold'] or low<settings['cohesion_threshold']):
            continue
        if any(r[2]!=owner and mean-max(r[0],r[5])<settings['ambiguity_margin'] for r in ranked):
            continue
        bank = search.bank.get(owner)
        if bank is None or (scan,row['cue']) in bank['cues']:
            continue
        if old and (old not in search.bank or search.bank[old]['faces']
                    or any(s!=scan for s,_ in search.bank[old]['cues'])):
            continue
        if _global_rival(search,query,{owner,old},mean-settings['ambiguity_margin']):
            continue
        proof = dict(person=owner, status='accepted', role='auxiliary',
                     reason='external_rear_view', source_scan=source, perspective='back', score=mean,
                     previous=dict(decision,person=old,status=row['status'],role=row['role']),
                     source_tracks=[[*w['roots'],w['id']] for w in witnesses])
        proposals[row['id']] = (owner,row['cue'],old,proof)
    # Freeze competing simultaneous claims and move a pre-existing fragment atomically.
    simultaneous = Counter((owner,cue) for owner,cue,_,_ in proposals.values())
    viable = {oid for oid,(owner,cue,_,_) in proposals.items() if simultaneous[owner,cue]==1}
    targets = {}
    for old,ids in required.items():
        if old not in face_owners and ids <= viable:
            destinations = {proposals[i][0] for i in ids}
            if len(destinations)==1:
                targets[old] = destinations.pop()
    for oid in sorted(viable):
        owner,_,old,proof = proposals[oid]
        if old and targets.get(old)!=owner:
            continue
        db.execute("UPDATE body_members SET person=?,status='accepted',role='auxiliary',proof=? WHERE observation=?",
                   (owner,encoded(proof),oid))
    for old,owner in targets.items():
        for oid,old_proof in satellites[old].items():
            if oid in viable and proposals[oid][0]==owner:
                continue
            proof = dict(person=None,status='withheld',role='auxiliary',reason='external_rear_satellite',
                         retry_when='confirmed_core_changes',retry_owner=owner,previous=old_proof)
            db.execute("UPDATE body_members SET person=NULL,status='withheld',role='auxiliary',proof=? WHERE observation=?",
                       (encoded(proof),oid))



def recover(db, selected, config):
    if not selected:
        return
    from .rear_gallery import load
    from .assignment_delta import enqueue
    scope,namespace=db.execute('SELECT scope,namespace FROM scans WHERE id=?',(selected[0]['scan'],)).fetchone()
    stamp = policy(config)
    # Snapshot only previously derived claims on disk, then restore their
    # frozen local decisions before taking a rival index or recomputing.
    db.execute("""CREATE TEMP TABLE rear_prior AS SELECT m.observation,b.scan,s.scene,m.person,m.proof
        FROM body_members m JOIN body_observations b ON b.id=m.observation JOIN scans s ON s.id=b.scan
        WHERE s.scope=? AND s.namespace=? AND json_extract(m.proof,'$.reason') IN ('external_rear_view','external_rear_satellite')
        AND """+predicate(),(scope,namespace))
    try:
        scans = {s['scan']:s['scene'] for s in selected}
        scans.update(db.execute('SELECT DISTINCT scan,scene FROM rear_prior'))
        for oid,raw in db.execute('SELECT observation,proof FROM rear_prior'):
            prior = json.loads(raw)['previous']
            db.execute('UPDATE body_members SET person=?,role=?,status=?,proof=? WHERE observation=?',
                       (prior['person'],prior['role'],prior['status'],encoded(prior),oid))
        if config.data['sapiensid']['global_matching']['enabled']:
            gallery = load(db,config,scope,namespace)
            if gallery.tree is not None:
                _recover_scans(db,scope,namespace,scans,config,stamp,gallery)
        changed = set(db.execute("""SELECT DISTINCT p.scan,p.scene FROM rear_prior p JOIN body_members m
            ON m.observation=p.observation WHERE m.person IS NOT p.person OR m.proof<>p.proof"""))
        # New assignments in selected scenes can change publication even when
        # the cached local stage was identical.
        for scan, scene in scans.items():
            active = db.execute("""SELECT 1 FROM body_members m JOIN body_observations b ON b.id=m.observation
                WHERE b.scan=? AND json_extract(m.proof,'$.reason')='external_rear_view' LIMIT 1""",(scan,)).fetchone()
            if active or (scan,scene) in changed:
                enqueue(db,scope,namespace,dict(scan=scan,scene=scene),config,True,'rear_view_attachment')
    finally:
        db.execute('DROP TABLE rear_prior')


def _recover_scans(db,scope,namespace,scans,config,stamp,gallery):
    from .body_observations import compatible_contracts
    from .body_global_banks import banks
    from .body_global_search import Search
    stale = db.execute("""SELECT 1 FROM body_members m JOIN body_observations b ON b.id=m.observation
        JOIN scans s ON s.id=b.scan WHERE s.scope=? AND s.namespace=? AND m.person IS NOT NULL
        AND m.role='anchor' AND b.contract NOT IN (SELECT value FROM json_each(?)) AND """+predicate()+
        ' LIMIT 1',(scope,namespace,encoded(compatible_contracts(config)))).fetchone()
    if stale:
        return
    references = banks(db,config,scope,namespace)
    if any(b['stale_policy'] or b['missing_original'] for b in references.values()):
        return
    with Search(db,config,references) as search:
        for scan in sorted(scans):
            _scene(db,scan,stamp,gallery,search,config)
