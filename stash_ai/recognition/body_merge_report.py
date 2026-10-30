"""Pair proposals are certified by production replay, including scene identity births."""
from .body_solver import analyse as hints
from .merge_report_policy import patched,proposal


def analyse(db,config,identities,scope,namespace,*,verify=True):
    from .merge_goals import resolve,evaluate
    goals=resolve(db,config,[list(identities)],scope=scope,by_performer=False)
    measured=hints(db,config,goals)
    result=dict(route='independent_person',current=measured['evidence'],attempts=[],recommendations=[])
    if not verify:
        result.update(verification_scope='measured_only', proposals=[
            dict(proposal(config,changes),strategy='measured_person_proposal',
                 verification=dict(replay_verified=False, shared_component=dict(people=list(identities),outcome='unverified')))
            for changes in [{},*measured['proposals']]])
        return result
    variants={};seen=set()
    from .config import digest
    for changes in [{},*measured['proposals']]:
        key=digest(changes)
        if key in seen:continue
        seen.add(key)
        if changes:variants['body-proposal-'+str(len(variants)+1)]=patched(config,changes)
    simulation=evaluate(config,goals,db,variants)
    for name,trial in simulation.get('trials',{}).items():
        settings=config if name=='baseline' else variants[name]
        from .threshold_tuning import BODY_PARAMETERS
        from .merge_report_policy import parameter
        changes={p:parameter(settings.data,p) for p in BODY_PARAMETERS if parameter(settings.data,p)!=parameter(config.data,p)}
        outcome=trial['goal_result'];group=outcome['groups'][0]
        item=dict(proposal(config,changes),strategy='production_person_replay',
            verification=dict(replay_verified=True,goal_result=outcome,
                shared_component=dict(people=list(identities)+group['additional_retained_identities'],outcome='merge' if outcome['fully_solved'] else 'blocked')))
        result['attempts'].append(item)
        if outcome['fully_solved']:result['recommendations'].append(item)
    return result
