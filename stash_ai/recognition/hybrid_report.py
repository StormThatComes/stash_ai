"""Measured hybrid proposals checked against the entire frozen competitor bank."""
from .hybrid_evidence import catalogue, pair, plan
from .merge_report_policy import _floor, patched, proposal
from .global_views import _cooccurs


def analyse(db, config, identities, scope, namespace):
    bank=catalogue(db,config,scope,namespace)
    by_id={x.person:x for x in bank}
    def check(settings):
        trial_bank=bank if settings is config else catalogue(db,settings,scope,namespace)
        result=plan(db,settings,trial_bank,focus=set(identities))
        shared=next((d for d in result['decisions'] if set(identities)<=set(d['people'])),None)
        blockers=shared['blockers'] if shared else ['no_shared_supported_component']
        return dict(can_merge_in_frozen_catalogue=not blockers,blockers=blockers,
                    shared_component=shared,components=result['decisions'],pair_evidence=result['pairs'])
    current=check(config)
    result=dict(route='sapiensid_hybrid',current=current,attempts=[],recommendations=[])
    if current['can_merge_in_frozen_catalogue']:
        result['recommendations'].append(dict(proposal(config,{}),verification=current))
        return result
    if any(i not in by_id for i in identities):
        result['unresolved_reasons']=['missing_strict_references']
        return result
    proof=pair(*(by_id[i] for i in identities),config.data['sapiensid']['matching'],
               copresent=_cooccurs(db,*identities,config),
               reflection_threshold=config.data['recognition']['cooccurrence_merge_threshold'])
    # Missing or insufficient evidence needs collection, never a fabricated vote.
    if any(b in proof['blockers'] for b in ('missing_person_evidence','insufficient_independent_support','missing_original_core_evidence')):
        result['unresolved_reasons']=proof['blockers']
        return result
    changes={}
    for key,measurement in (('face_floor',proof['face']),('person_threshold',proof['person'])):
        if measurement is None or measurement['minimum']<0:
            result['unresolved_reasons']=['negative_or_missing_similarity']
            return result
        changes['sapiensid.matching.'+key]=min(config.data['sapiensid']['matching'][key],_floor(measurement['minimum']))
    if proof['cooccurring']:
        changes['recognition.cooccurrence_merge_threshold']=min(
            config.data['recognition']['cooccurrence_merge_threshold'],_floor(proof['face']['minimum']))
    verification=check(patched(config,changes))
    # Never reduce ambiguity margin to erase an observed competitor.
    item=dict(proposal(config,changes),verification=verification,strategy='measured_hybrid_gates')
    result['attempts'].append(item)
    if verification['can_merge_in_frozen_catalogue']:
        result['recommendations'].append(item)
    return result
