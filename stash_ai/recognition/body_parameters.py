"""Retained body matching parameters shared by tuning and solver exploration."""
from .person_anchor import PIXEL_PARAMETER

LOCAL = tuple('sapiensid.identities.'+key for key in (
    'match_threshold','cohesion_threshold','ambiguity_margin','min_support',
    'attach_min_support','min_appearance_distance','anchor_min_keypoints','anchor_min_sharpness','anchor_min_person_pixels'))
GLOBAL = tuple('sapiensid.global_matching.'+key for key in (
    'match_threshold','cohesion_threshold','ambiguity_margin','min_support'))
QUALITY=tuple('sapiensid.identities.'+key for key in ('anchor_min_keypoints','anchor_min_sharpness'))
PARAMETERS = frozenset(LOCAL+GLOBAL)


def active(config):
    from .body_identity import enabled
    settings=config.data['sapiensid']
    # Null is an explicit opt-out, like the optional face admission gates.
    pixel=(PIXEL_PARAMETER,) if settings['enabled'] and settings['identities']['anchor_min_person_pixels'] is not None else ()
    if not settings['enabled']:return ()
    if not enabled(config):return QUALITY+pixel
    return tuple(k for k in LOCAL if k!=PIXEL_PARAMETER)+pixel+(GLOBAL if settings['global_matching']['enabled'] else ())
