"""Measured independent-person hints; production replay validates every trial."""
from .body_identity import enabled,available
from .body_global import report
from .merge_report_policy import _floor
from .threshold_boundaries import sides
from .body_parameters import active


def options_and_quality(db,config,goals):
    """Numerical proposals use retained measurements, never new model inputs."""
    import json
    from .body_observations import contract
    from .database import encoded
    from .merge_report_policy import parameter
    from .merge_solver_domains import registry
    from .merge_solver_search import gate_values
    from .merge_solver_options import compact
    from .scan_heads import predicate as current_scan
    domains=registry(config)
    options={key:list(gate_values(key,parameter(config.data,key),(),domain=domains[key])) for key in active(config)}
    # dHash distances are small integers. Every valid boundary is cheap to
    # explore, without building a quadratic matrix of all retained crops.
    options['sapiensid.identities.min_appearance_distance']=list(range(65))
    people=sorted({m['identity_id'] for g in goals['groups'] for m in g['members']})
    rows=db.execute('SELECT b.quality FROM body_observations b JOIN body_members m ON m.observation=b.id '
        'JOIN scans s ON s.id=b.scan WHERE m.person IN (SELECT value FROM json_each(?)) '
        "AND s.scope=? AND s.namespace=? AND s.status='complete' AND b.contract=? "
        'AND b.rejected IS NULL AND b.embedding IS NOT NULL AND '+current_scan()+' ORDER BY b.id',
        (encoded(people),goals['scope'],goals['namespace'],contract(config)))
    joint={}
    for row in rows:
        quality=json.loads(row['quality']).get('quality',{})
        for field,key in (('keypoints','sapiensid.identities.anchor_min_keypoints'),
                          ('sharpness','sapiensid.identities.anchor_min_sharpness')):
            value=quality.get(field);domain=domains[key]
            if type(value) not in (int,float) or not domain.low<=value<=domain.high:continue
            values=([int(value),int(value)+1] if domain.kind is int else sides(value,upper=domain.high))
            options[key].extend(domain.project(v) for v in values)
            if value<parameter(config.data,key):joint[key]=min(joint.get(key,value),value)
        if any(len(values)>256 for values in options.values()):options=compact(config,{},options)
    return compact(config,{},options),([joint] if joint else [])


def analyse(db,config,goals):
    output=dict(options={},proposals=[],evidence=dict(enabled=False,pairs=[]))
    if not enabled(config) or not available(db):return output
    owners={m['identity_id']:g.get('index',i) for i,g in enumerate(goals['groups']) for m in g['members']}
    evidence=report(db,config,goals['scope'],goals['namespace'],focus=owners)
    output['evidence']=evidence
    output['options'],quality_proposals=options_and_quality(db,config,goals)
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
