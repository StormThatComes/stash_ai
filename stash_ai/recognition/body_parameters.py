"""Retained body matching parameters shared by tuning and solver exploration."""
LOCAL = tuple('sapiensid.identities.'+key for key in (
    'match_threshold','cohesion_threshold','ambiguity_margin','min_support',
    'attach_min_support','min_appearance_distance','anchor_min_keypoints','anchor_min_sharpness'))
GLOBAL = tuple('sapiensid.global_matching.'+key for key in (
    'match_threshold','cohesion_threshold','ambiguity_margin','min_support'))
PARAMETERS = frozenset(LOCAL+GLOBAL)


def active(config):
    from .body_identity import enabled
    if not enabled(config):return ()
    return LOCAL+(GLOBAL if config.data['sapiensid']['global_matching']['enabled'] else ())
