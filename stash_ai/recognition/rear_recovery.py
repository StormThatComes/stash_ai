"""Attach faceless rear queries to certified external views in one generation."""
from collections import Counter, defaultdict
import json
import numpy as np

from .body_identity import policy, quality, vector, scene_bonus
from .database import encoded
from .scan_heads import predicate


def _global_rivals(search, batch, *, plausible=None):
    """Read each centroid tile once for all bounded queries."""
    if not batch:
        return {}
    ids, queries, ignored, floors = zip(*batch)
    queries = np.stack(queries, axis=1).astype(np.float64)
    rivals = {}
    bounds = getattr(search,'rear_bounds',None)
    if bounds is None and hasattr(search,'stack') and len(search.entries)>search.block:
        from .rear_bounds import RivalBounds
        search.rear_bounds = bounds = RivalBounds(search)
    norms = np.linalg.norm(queries,axis=0)
    floors = np.asarray(floors)
    for start in range(0, len(search.entries), search.block):
        if bounds is not None:
            possible = bounds.possible(start//search.block,queries,norms,floors)
            if not any(possible[col] and oid not in rivals for col,oid in enumerate(ids)):
                search.stats['rear_rival_blocks_skipped'] = search.stats.get('rear_rival_blocks_skipped',0)+1
                continue
        count = min(search.block, len(search.entries)-start)
        search.index.seek(start*4096*8)
        tile = np.frombuffer(search.index.read(count*4096*8), dtype='<f8').reshape(count,4096)
        scores = tile @ queries
        for col, oid in enumerate(ids):
            if oid in rivals:
                continue
            column = scores[:,col]
            if np.any(np.abs(column + 1e-6 - floors[col]) <= 1e-8):
                column = tile @ np.ascontiguousarray(queries[:,col])
            for offset in np.flatnonzero(column+1e-6 >= floors[col]):
                owner = search.identities[search.entries[start+int(offset)][0]]
                if owner not in ignored[col] and (plausible is None or plausible(oid,owner,search.entries[start+int(offset)][1])):
                    rivals[oid] = owner
                    break
        if len(rivals) == len(batch):
            break
    return rivals


def _rows(db, scan, stamp):
    return db.execute("""SELECT b.id,b.cue,b.seconds,b.scan,b.face,b.quality,b.geometry,b.embedding,
        m.person,m.status,m.role,m.proof,f.rejected AS face_rejection,f.embedding AS face_embedding,
        f.quality AS face_quality,f.dimensions AS face_dimensions,fm.person AS face_owner FROM body_observations b
        LEFT JOIN observations f ON f.id=b.face LEFT JOIN members fm ON fm.observation=f.id JOIN body_members m
        ON m.observation=b.id WHERE b.scan=? AND m.policy=? AND b.rejected IS NULL
        ORDER BY b.cue,b.id""",(scan,stamp))


def _occupied_torsos(db, scan, stamp, config):
    from .pose_policy import anchor_sql
    from .reference_roles import strict_sql
    # Only trusted face/body anchors can exclude a physically distinct rival.
    source = """SELECT b.id,b.cue,b.geometry,m.person FROM body_observations b
        JOIN body_members m ON m.observation=b.id JOIN observations o ON o.id=b.face
        JOIN members f ON f.observation=o.id AND f.person=m.person
        WHERE b.scan=? AND m.policy=? AND m.role='anchor' AND m.status='accepted'
        AND o.rejected IS NULL AND b.rejected IS NULL AND """ + strict_sql('o','f') + ' AND '+anchor_sql('o',config.data['clustering'])
    torsos = defaultdict(list)
    for oid,cue,geometry,owner in db.execute(source,(scan,stamp)):
        torsos[cue].append((oid,owner,json.loads(geometry)))
    return torsos


def impossible_owners(row,person,torsos):
    from .crowd_context import pair
    return {owner for oid,owner,other in torsos if oid!=row['id'] and pair([person,other]) is not None}


def _scene(db, scan, stamp, gallery, search, config):
    from .body_perspectives import orientation
    settings = config.data['sapiensid']['identities']
    proposals = {}
    required = defaultdict(set)
    satellites = defaultdict(dict)
    face_owners = set()
    anonymous = set()
    nonanonymous = set()
    core_rows = defaultdict(list)
    from .body_guards import FaceGuard
    from .clustering import vector as face_vector
    from .rear_gallery import winner
    guard = FaceGuard(db, config)
    batch = []
    torsos = _occupied_torsos(db,scan,stamp,config)
    waiting = {}
    decisions = {}
    query_faces = {}
    checked = defaultdict(set)
    def plausible(oid,owner,refs):
        faces = query_faces.get(oid)
        # Incomplete or historical banks stay conservative competitors.
        if not faces or owner in search.unavailable or not refs or len(checked[oid])>=8:
            return True
        if owner in checked[oid]:
            return False
        if any(guard.allows({'_face_vector':face},(),owner) for face in faces):
            return True
        checked[oid].add(owner)
        return False

    def verify_batch():
        for oid, rival in _global_rivals(search, batch,plausible=plausible).items():
            _, _, _, proof = proposals.pop(oid)
            waiting[oid] = (proof['previous'], rival, 'competing_owner')
        for oid,owners in checked.items():
            if oid in proposals and owners:
                proposals[oid][3]['checked_rivals'] = sorted(owners)
        batch.clear()
        checked.clear()
        query_faces.clear()

    def request(row,decision):
        old = row['person']
        if (row['face_owner'] or row['status']=='accepted' and decision.get('reason') not in ('body_scene_core','body_core_attachment')
                or decision.get('track_blocker',{}).get('reason') in ('competing_transition','different_face_root','unresolved_partner')):
            return None
        person = json.loads(row['geometry'])
        kind = orientation(person) if person.get('keypoints') else quality(row).get('orientation')
        kind = decision.get('query_perspective') or kind
        if kind not in ('back','side'):
            return None
        query = vector(row)
        context = decision.get('shot_context')
        roster = set(decision.get('scene_roster',()))
        rooted_owner = decision.get('open_face_root',{}).get('owner')
        if context and context['owner']==rooted_owner:
            roster.add(rooted_owner)
        bonus = scene_bonus(config.data) if roster else 0.
        search_settings = dict(settings,match_threshold=max(0.,settings['match_threshold']-bonus))
        return query,scan,search_settings,quality(row).get('pixel_digest'),kind

    def queries():
        from itertools import islice
        source = iter(_rows(db,scan,stamp))
        size = min(16,search.block) if hasattr(gallery,'rank_many') else 1
        while batch_rows := list(islice(source,size)):
            items, requests = [], []
            for raw in batch_rows:
                row = dict(raw)
                decision = json.loads(row['proof'])
                decision.update(person=row['person'],status=row['status'],role=row['role'])
                decision.pop('rear_retry',None)
                query = request(row,decision)
                if query is not None:
                    requests.append(query)
                items.append((row,decision,query))
            ranks = iter(gallery.rank_many(requests) if hasattr(gallery,'rank_many') else
                         [gallery.rank(*r) for r in requests])
            for row,decision,query in items:
                yield row,decision,query,next(ranks) if query is not None else None

    for row,decision,request_args,ranked in queries():
        old = row['person']
        if old is not None:
            if row['face_owner'] or row['face_rejection'] is None and row['face_embedding'] is not None:
                face_owners.add(old)
            if decision.get('reason')=='anonymous_body_track':
                anonymous.add(old)
            else:
                nonanonymous.add(old)
            if row['role']=='anchor' or decision.get('reason') in ('body_scene_core','anonymous_body_track'):
                if settings.get('tracklet_matching'):
                    core_rows[old].append({key:row[key] for key in ('id','scan','cue','seconds','quality','geometry')})
                required[old].add(row['id'])
            else:
                satellites[old][row['id']] = dict(decision,person=old,status=row['status'],role=row['role'])
        if request_args is None:
            continue
        query,_,search_settings,_,kind = request_args
        context = decision.get('shot_context')
        roster = set(decision.get('scene_roster',()))
        if context and context['owner']==decision.get('open_face_root',{}).get('owner'):
            roster.add(context['owner'])
        bonus = settings['match_threshold']-search_settings['match_threshold']
        occupied = impossible_owners(row,json.loads(row['geometry']),torsos.get(row['cue'],()))
        if occupied:
            ranked = [candidate for candidate in ranked if candidate[2] not in occupied]
        from .body_perspectives import owner_bonus
        bonuses = {owner:owner_bonus(decision,owner,bonus) for owner in roster}
        best = winner(ranked,settings,owner_bonuses=bonuses)
        endpoint = None
        if best is None and hasattr(search,'endpoints'):
            def reject_rival(rival,faces):
                if rival in search.unavailable or rival not in search.bank or not search.bank[rival]['faces']:
                    return False
                if all(not guard.allows({'_face_vector':face},(),rival) for face,_ in faces):
                    checked[row['id']].add(rival)
                    return True
                return False
            endpoint = search.endpoints.match(scan,kind,decision,occupied,ranked,reject_rival=reject_rival)
        if best is None and endpoint is None:
            if ranked:
                waiting[row['id']] = (decision, ranked[0][2], 'no_separated_view')
            continue
        if endpoint:
            owner,face_score,witness = endpoint
            source,witnesses = witness['scan'],[witness]
            # Keep body competition on the ordinary body scale.
            mean,low = settings['match_threshold'],settings['cohesion_threshold']
        else:
            mean,low,owner,source,witnesses,_ = best
        if (owner==old or decision.get('open_face_root',{}).get('owner',owner)!=owner
                or low<settings['cohesion_threshold']):
            continue
        if row['face_embedding'] is not None and row['face_rejection'] is None:
            measured = face_vector(dict(dimensions=row['face_dimensions'], embedding=row['face_embedding']))
            if not guard.allows({'_face_vector':measured,'face_quality':row['face_quality']}, (), owner):
                continue
        bank = search.bank.get(owner)
        if (bank is None or owner in search.unavailable or old in search.unavailable
                or (scan,row['cue']) in bank['cues']):
            continue
        if old and (old not in search.bank and decision.get('reason')!='anonymous_body_track' or
                    old in search.bank and (search.bank[old]['faces'] or any(s!=scan for s,_ in search.bank[old]['cues']))):
            continue
        proof = dict(person=owner, status='accepted', role='auxiliary',
                     reason='external_rear_view', source_scan=source, perspective=kind, score=mean,
                     previous=dict(decision,person=old,status=row['status'],role=row['role']),
                     source_tracks=[[*w['roots'],w['id']] for w in witnesses],
                     source_scans=sorted({w['scan'] for w in witnesses}))
        if endpoint:
            proof.pop('score')
            proof.update(route='fragment_face_endpoints',face_score=face_score,
                         local_roots=decision['rooted_track']['roots'])
        if mean<settings['match_threshold']:
            if context and context['owner']==owner:
                proof['shot_context'] = context
            else:
                proof['roster_context'] = dict(owner=owner,bonus=bonuses.get(owner,0.),
                    witnesses=decision.get('roster_witnesses',{}).get(owner,[]))
        competitors = [(max(item[0],item[5]),item[2]) for item in ranked if item[2]!=owner]
        if competitors:
            runner,rival = max(competitors)
            proof.update(runner=runner,rival=rival,rival_scope='certified_rear_views')
        if row['face_embedding'] is not None and row['face_rejection'] is None:
            query_faces[row['id']] = [measured]
        elif hasattr(search,'endpoints'):
            track = decision.get('rooted_track') or {}
            if track.get('owner')==owner and len(track.get('roots',()))==2:
                ends = [search.endpoints.face(root,owner) for root in track['roots']]
                if all(ends) and ends[0][1]!=ends[1][1]:
                    query_faces[row['id']] = [entry[0] for entry in ends]
        proposals[row['id']] = (owner,row['cue'],old,proof)
        batch.append((row['id'], query, {owner,old} | occupied, mean-settings['ambiguity_margin']))
        if len(batch) >= min(search.block, 32):
            verify_batch()
    verify_batch()
    for oid, (previous, rival, reason) in waiting.items():
        proof = dict(previous, rear_retry=dict(reason=reason, rival=rival, references=gallery.revision))
        decisions[oid] = proof
    # Freeze competing simultaneous claims and move a pre-existing fragment atomically.
    simultaneous = Counter((owner,cue) for owner,cue,_,_ in proposals.values())
    viable = {oid for oid,(owner,cue,_,_) in proposals.items() if simultaneous[owner,cue]==1}
    targets = {}
    for old,ids in required.items():
        if old not in face_owners and ids <= viable:
            destinations = {proposals[i][0] for i in ids}
            if len(destinations)==1:
                targets[old] = destinations.pop()
    split = set()
    if settings.get('tracklet_matching'):
        from .body_tracklets import split_members
        proposed = {oid:proposals[oid][0] for oid in viable}
        def occupied(row,target):
            return target in impossible_owners(row,json.loads(row['geometry']),torsos.get(row['cue'],()))
        for old,core in core_rows.items():
            if old not in targets and old not in face_owners:
                allowed = split_members(core,proposed,occupied,settings)
                split.update(allowed)
                if len({proposed[oid] for oid in allowed})==2 and old not in (anonymous & nonanonymous):
                    for row in core:
                        if row['id'] in allowed:
                            continue
                        prior = db.execute('SELECT person,role,status,proof FROM body_members WHERE observation=?',
                                           (row['id'],)).fetchone()
                        decisions[row['id']] = dict(person=None,status='withheld',role='auxiliary',
                            reason='external_rear_satellite',track_blocker=dict(reason='unresolved_partner',cue=row['cue']),
                            previous=dict(json.loads(prior['proof']),person=prior['person'],role=prior['role'],status=prior['status']))
    mixed_origins = anonymous & nonanonymous
    for oid in sorted(viable):
        owner,_,old,proof = proposals[oid]
        if old in mixed_origins or (old and targets.get(old)!=owner and oid not in split):
            continue
        decisions[oid] = proof
    for old,owner in targets.items():
        for oid,old_proof in satellites[old].items():
            if oid in viable and proposals[oid][0]==owner:
                continue
            proof = dict(person=None,status='withheld',role='auxiliary',reason='external_rear_satellite',
                         retry_when='confirmed_core_changes',retry_owner=owner,previous=old_proof)
            decisions[oid] = proof
    return decisions



def recover(db, selected, config):
    if not selected:
        return
    from .rear_gallery import load
    from .assignment_delta import enqueue
    scope,namespace=db.execute('SELECT scope,namespace FROM scans WHERE id=?',(selected[0]['scan'],)).fetchone()
    stamp = policy(config)
    # Present the frozen local memberships to all reference readers. Derived
    # claims remain durable until we have a verified replacement for their scan.
    from .rear_snapshot import frozen, apply
    from contextlib import ExitStack, closing
    from .scratch_space import scope as scratch_scope, connect
    with ExitStack() as resources, frozen(db,scope,namespace) as active:
        scans = {s['scan']:s['scene'] for s in selected}
        gallery = None
        if config.data['sapiensid']['global_matching']['enabled']:
            pages = None
            if config.data['execution']['rebuild_storage']=='disk':
                resources.enter_context(scratch_scope(config))
                pages = resources.enter_context(closing(connect()))
            gallery = load(db,config,scope,namespace,pages=pages)
        if gallery and gallery.tree is not None and _recover_scans(db,scope,namespace,scans,config,stamp,gallery,active):
            return
        # Incompatible or withdrawn evidence revokes previously derived claims.
        from .config import digest
        for scan,scene in active.items():
            db.execute('DELETE FROM metadata WHERE key=?',('rear-pending:'+digest([scope,namespace,scene]),))
            if apply(db,scan,{}):
                enqueue(db,scope,namespace,dict(scan=scan,scene=scene),config,True,'rear_view_attachment')


def _recover_scans(db,scope,namespace,scans,config,stamp,gallery,active):
    from .body_observations import compatible_contracts
    from .body_global_banks import _banks
    from .body_global_search import Search
    from .rear_snapshot import apply
    from .assignment_delta import enqueue
    stale = db.execute("""SELECT DISTINCT m.person FROM body_members m JOIN body_observations b ON b.id=m.observation
        JOIN scans s ON s.id=b.scan WHERE s.scope=? AND s.namespace=? AND m.person IS NOT NULL
        AND m.role='anchor' AND b.contract NOT IN (SELECT value FROM json_each(?)) AND """+predicate(),(scope,namespace,encoded(compatible_contracts(config))))
    from .identity_merges import canonical
    unavailable = {canonical(db, row[0]) for row in stale}
    references = _banks(db,config,scope,namespace)
    unavailable.update(owner for owner, bank in references.items()
                       if bank['stale_policy'] or bank['missing_original'])
    from .rear_pending import Pending
    from .rear_gallery import revision
    gallery.revision = revision(gallery)
    with Search(db,config,references) as search:
        search.unavailable = unavailable
        from .fragment_endpoints import enabled as endpoints_enabled
        if endpoints_enabled(config.data['sapiensid']['identities']):
            from .fragment_endpoints import ExternalEndpoints
            search.endpoints = ExternalEndpoints(db,gallery,config)
        settings = config.data['sapiensid']['identities']
        minimum = max(0.,settings['match_threshold'] - scene_bonus(config.data))
        pending = Pending(db,gallery,search,dict(settings,match_threshold=minimum))
        for scan,scene in pending.candidates(scope,namespace,set(scans),stamp,active=active):
            decisions = _scene(db,scan,stamp,gallery,search,config)
            if apply(db,scan,decisions):
                enqueue(db,scope,namespace,dict(scan=scan,scene=scene),config,True,'rear_view_attachment')
    return True
