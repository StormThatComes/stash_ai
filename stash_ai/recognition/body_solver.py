"""Measured independent-person hints; production replay validates every trial."""
from .body_identity import enabled,available
from .body_global import report
from .merge_report_policy import _floor
from .threshold_boundaries import sides
from .body_parameters import active


def options_and_quality(db,config,goals,*,frozen_goals=None):
    """Numerical proposals use retained measurements, never new model inputs."""
    from .body_observations import contract
    from .database import encoded
    from .merge_report_policy import parameter
    from .merge_solver_domains import registry
    from .merge_solver_search import gate_values
    from .merge_solver_options import compact
    from .scan_heads import predicate as current_scan
    from .person_anchor import anchor_gate,PIXEL_PARAMETER
    from .person_signal import contract as person_contract,available as persons_available
    from itertools import chain
    domains=registry(config)
    options={key:list(gate_values(key,parameter(config.data,key),(),domain=domains[key])) for key in active(config)}
    # dHash distances are small integers. Every valid boundary is cheap to
    # explore, without building a quadratic matrix of all retained crops.
    if 'sapiensid.identities.min_appearance_distance' in options:
        options['sapiensid.identities.min_appearance_distance']=list(range(65))
    people=sorted({m['identity_id'] for g in goals['groups'] for m in g['members']})
    ids=sorted({r['id'] for g in (frozen_goals or {'groups':[]})['groups'] for m in g['members'] for r in m.get('observations',())})
    body_rows=db.execute('SELECT b.id,b.quality FROM body_observations b LEFT JOIN body_members m ON m.observation=b.id '
        'JOIN scans s ON s.id=b.scan WHERE (m.person IN (SELECT value FROM json_each(?)) '
        'OR b.id IN (SELECT value FROM json_each(?)) OR b.face IN (SELECT value FROM json_each(?))) '
        "AND s.scope=? AND s.namespace=? AND s.status='complete' AND b.contract=? "
        'AND b.rejected IS NULL AND b.embedding IS NOT NULL AND '+current_scan()+' ORDER BY b.id',
        (encoded(people),encoded(ids),encoded(ids),goals['scope'],goals['namespace'],contract(config))) if enabled(config) and available(db) else ()
    face_rows=db.execute('SELECT o.id,e.metadata AS quality FROM person_evidence e JOIN observations o ON o.id=e.observation '
        'LEFT JOIN members m ON m.observation=o.id JOIN scans s ON s.id=o.scan '
        'WHERE (m.person IN (SELECT value FROM json_each(?)) OR o.id IN (SELECT value FROM json_each(?))) AND s.scope=? AND s.namespace=? '
        "AND s.status='complete' AND o.rejected IS NULL AND e.contract=? AND e.status='ready' AND "+current_scan(),
        (encoded(people),encoded(ids),goals['scope'],goals['namespace'],person_contract(config))) if options and config.data['sapiensid']['enabled'] and persons_available(db) else ()
    joint={};measurements=[]
    for row in chain(body_rows,face_rows):
        measured=anchor_gate(row['quality'],config.data['sapiensid']['identities'],
            collection=config.data['sapiensid']['collection'])
        patch={}
        for field,key in (('keypoints','sapiensid.identities.anchor_min_keypoints'),
                          ('sharpness','sapiensid.identities.anchor_min_sharpness'),('pixels',PIXEL_PARAMETER)):
            if key not in options:continue
            value=measured.get(field);domain=domains[key]
            if type(value) not in (int,float) or not domain.low<=value<=domain.high:continue
            values=([int(value),int(value)+1] if domain.kind is int else sides(value,upper=domain.high))
            options[key].extend(domain.project(v) for v in values)
            if value<parameter(config.data,key):
                admitted=domain.project(int(value) if domain.kind is int else value)
                joint[key]=min(joint.get(key,admitted),admitted)
                patch[key]=admitted
        if patch and frozen_goals is not None:measurements.append(dict(observation=row['id'],patch=patch))
        if any(len(values)>256 for values in options.values()):options=compact(config,{},options)
    return compact(config,{},options),([joint] if joint else []),measurements


def analyse(db,config,goals,*,frozen_goals=None):
    output=dict(options={},proposals=[],quality=[],evidence=dict(enabled=False,pairs=[]))
    if not active(config):return output
    output['options'],quality_proposals,output['quality']=options_and_quality(db,config,goals,frozen_goals=frozen_goals)
    output['proposals']=quality_proposals
    if not enabled(config) or not available(db):return output
    owners={m['identity_id']:g.get('index',i) for i,g in enumerate(goals['groups']) for m in g['members']}
    evidence=report(db,config,goals['scope'],goals['namespace'],focus=owners)
    output['evidence']=evidence
    output['proposals']=[]
    proposals=[]
    for pair in evidence['pairs']:
        a,b=pair['people']
        if a not in owners and b not in owners:continue
        pair['relation']='within_set' if a in owners and b in owners and owners[a]==owners[b] else 'competitor'
        prefix='sapiensid.global_matching.'
        for key,field in (('match_threshold','score'),('cohesion_threshold','minimum')):
            output['options'].setdefault(prefix+key,[]).extend(sides(pair[field]))
        if pair['relation']!='within_set' or min(pair['score'],pair['minimum'])<0:continue
        if set(pair['blockers']) & {'person_copresence','face_contradiction','missing_original_body_witness'}:continue
        settings=config.data['sapiensid']['global_matching']
        patch={prefix+'match_threshold':min(settings['match_threshold'],_floor(pair['score'])),
               prefix+'cohesion_threshold':min(settings['cohesion_threshold'],_floor(pair['minimum'])),
               prefix+'min_support':min(settings['min_support'],min(pair['support']))}
        rivals=[p['score'] for p in evidence['pairs'] if set(p['people']) & {a,b} and
                any(x not in owners or owners.get(x)!=owners[a] for x in p['people'])]
        rivals.extend(p['score'] for p in evidence.get('nearest_competitors',()) if p['identity'] in (a,b))
        if rivals and pair['score']>=max(rivals):
            patch[prefix+'ambiguity_margin']=min(settings['ambiguity_margin'],_floor(pair['score']-max(rivals)))
        proposals.append(patch)
    if proposals:
        joint={}
        for patch in proposals:
            for key,value in patch.items():joint[key]=min(joint.get(key,value),value)
        output['proposals']=[joint,*proposals]
    output['proposals'].extend(quality_proposals)
    return output
