"""Abstain from publishing a new identity near a frozen existing appearance."""
from .clustering import vector
from .scoring import comparisons
from .view_prototypes import pose_matches
from .database import encoded


def potential(rows,history,config):
    settings,grouping=config.data['recognition'],config.data['clustering']
    threshold=max(settings['match_threshold'],grouping['view_match_threshold'])
    margin=max(settings['ambiguity_margin'],grouping['view_ambiguity_margin'])
    from .historical_references import _rank_bank
    groups, views = _rank_bank(history, config, False)
    candidates={}
    for row in rows:
        choices={};query=vector(row)
        for group,score,low in comparisons(history.groups,query,settings['match_threshold']-settings['ambiguity_margin'],grouping['cohesion_threshold']):
            if low>=grouping['cohesion_threshold'] and score>=settings['match_threshold']-settings['ambiguity_margin']:
                if score>choices.get(group.identifier,(float('-inf'),))[0]:
                    choices[group.identifier]=(score,'strict',settings['match_threshold'])
        for group,score,low in comparisons(groups,query,threshold-margin,grouping['view_cohesion_threshold']):
            view=views[id(group)]
            if low>=grouping['view_cohesion_threshold'] and score>=threshold-margin and pose_matches(row['geometry'],view.pose):
                if score>choices.get(group.identifier,(float('-inf'),))[0]:
                    choices[group.identifier]=(score,'view',threshold)
        for person,value in choices.items():candidates.setdefault(person,[]).append(value)
    rivals=[dict(identity=person,score=sum(v[0] for v in values)/len(values),
                 authority=sorted({v[1] for v in values}))
            for person,values in candidates.items() if len(values)==len(rows)]
    if not rivals:return None
    return dict(format=1,frozen=True,outcome='plausible_existing_identity',context=history.context,
                rivals=sorted(rivals,key=lambda r:(-r['score'],r['identity']))[:2],
                validation='whole_frozen_query_cohort',match_threshold=settings['match_threshold'],
                view_match_threshold=threshold)


def defer(db,source,proof):
    db.execute("UPDATE candidates SET person=NULL,status='withheld',payload=json_set(payload,"
        "'$.prior_reason',json_extract(payload,'$.reason'),'$.reason','historical_potential_identity',"
        "'$.global_identity_decision',json(?)) WHERE person=?",(encoded(proof),source))
    db.execute('DELETE FROM members WHERE person=?',(source,))
