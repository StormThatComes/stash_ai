"""Query-only recovery after independent scene cores obtain global support."""
from .resource_errors import ResourceLimit
import json
from .database import encoded


def recover(database,selected,config):
    from .body_identity import Scores,anchor,vector,policy
    from .body_observations import contract
    from .body_guards import FaceGuard,resolve_claims
    from .clustering import vector as face_vector
    db=database.connection;settings=config.data['sapiensid']['identities']
    reasons={'body_anchor_quality','body_anchor_pixels','body_anchor_pixels_missing',
             'body_anchor_keypoint_rules', 'body_anchor_keypoint_rules_missing', 'fragment_auxiliary_retry'}
    for item in selected:
        if not db.execute("SELECT 1 FROM body_members m JOIN body_observations b ON b.id=m.observation "
                "WHERE b.scan=? AND m.status='withheld' AND (json_extract(m.proof,'$.reason') IN "
                "(SELECT value FROM json_each(?)) OR json_extract(m.proof,'$.retry_when')='confirmed_core_changes') LIMIT 1",(item['scan'],encoded(sorted(reasons)))).fetchone():continue
        rows=[];decisions={};cost=0
        for raw in db.execute('SELECT b.*,m.person,m.status,m.role,m.proof,o.dimensions AS face_dimensions,'
                'o.quality AS face_quality,o.embedding AS face_embedding,o.rejected AS face_rejected FROM body_observations b '
                'JOIN body_members m ON m.observation=b.id LEFT JOIN observations o ON o.id=b.face '
                'WHERE b.scan=? AND b.contract=? AND b.rejected IS NULL AND b.embedding IS NOT NULL AND m.policy=?',
                (item['scan'],contract(config),policy(config))):
            row=dict(raw);cost+=len(row['embedding'])*3+len(row['quality'])+len(row['proof'])+4096
            if cost>config.data['sapiensid']['execution']['reference_bytes']:
                raise ResourceLimit('Person recovery references exceed sapiensid.execution.reference_bytes.')
            row['_vector']=vector(row)
            row['_face_vector']=(face_vector(dict(dimensions=row['face_dimensions'],embedding=row['face_embedding']))
                if row['face_embedding'] is not None and row['face_rejected'] is None else None)
            decisions[row['id']]=dict(json.loads(row['proof']),person=row['person'],status=row['status'],role=row['role'])
            rows.append(row)
        # Original scene anchors remain frozen, including those downgraded by
        # a global alias. Recovered queries can never enter these banks.
        banks={};confirmed=set();queries=[]
        for row in rows:
            proof=decisions[row['id']]
            if row['person'] is not None and not proof.get('view_exclusion') and anchor(row,settings) and (
                    row['role']=='anchor' or proof.get('reason')=='body_scene_core'):
                banks.setdefault(row['person'],[]).append(row)
                if row['status']=='accepted':confirmed.add(row['person'])
            elif row['status']=='withheld' and (proof.get('retry_when')=='confirmed_core_changes' or
                    proof.get('reason') in reasons and not anchor(row,settings)):
                queries.append(row)
        scores=Scores(config);guard=FaceGuard(db,config,scores=scores)
        from .body_seed_modes import FrozenSeeds
        modes = FrozenSeeds(banks,scores,settings) if queries else None
        for row in queries:
            prior = decisions[row['id']]
            blocker = prior.get('track_blocker', {}).get('reason')
            if blocker in ('different_face_root','unresolved_partner','competing_transition'):
                continue
            rooted_owner = prior.get('open_face_root', {}).get('owner')
            ranking = modes.rank(row,guard)
            allowed = [item for item in ranking if item[4] and item[2] in confirmed]
            if not allowed:
                continue
            mean,minimum,person,*_ = allowed[0]
            runner = max((item[5] for item in ranking if item[2]!=person),default=-1.)
            if (rooted_owner in (None,person) and person in confirmed and mean>=settings['match_threshold'] and minimum>=settings['cohesion_threshold']
                    and mean-runner>=settings['ambiguity_margin']):
                decisions[row['id']].update(person=person,status='accepted',role='auxiliary',
                    reason='body_confirmed_core_attachment',score=mean,minimum=minimum,runner=runner)
        before = {r['id']:(r['person'],r['status']) for r in rows}
        resolve_claims(rows,decisions,config,scores=scores)
        for row in rows:
            proof=decisions[row['id']]
            if before[row['id']]==(proof['person'],proof['status']):
                continue
            db.execute('UPDATE body_members SET person=?,status=?,role=?,proof=? WHERE observation=?',
                (proof['person'],proof['status'],proof['role'],encoded(proof),row['id']))
