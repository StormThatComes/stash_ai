"""Per-joint provenance stays separate from backend confidence semantics."""


def states(usable, person, record):
    conflicted=set(record.get('joint_ownership_withheld',())) | set(record.get('withheld_rival_joints',())) | set(record.get('mask_ownership_withheld',()))
    restored=set(record.get('restored_first_pass_joints',()))
    temporal={k for k,_ in record.get('temporal_prompts',())}
    visibility=person.get('keypoint_visibility')
    result=[]
    for k,valid in enumerate(usable):
        state='measured' if valid else 'unavailable'
        if valid and record.get('status')=='refined' and k not in restored:
            state='recovered' if k in temporal or record.get('contact_pose_recovered') else 'masked'
        if not valid and visibility is not None and visibility[k]<=.5:state='occluded'
        if k in conflicted:state='conflicted'
        result.append(state)
    return result
