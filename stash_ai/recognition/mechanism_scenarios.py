"""Shared retained-evidence experiments for the solver and pair inspector.

Plans contain YAML changes and explanations only. Recognition still constructs
all references, resolves competitors and assigns observations during replay.
"""
from .merge_report_policy import parameter
from .pose_policy import restricted
from .threshold_boundaries import at_or_above


INHERITED_SWITCHES = frozenset('sapiensid.identities.'+key for key in ('extended_track_graph','turn_recovery'))
SWITCHES = frozenset('clustering.' + key for key in (
    'auxiliary_matching', 'view_prototypes_enabled', 'context_view_enrichment',
    'corroborated_view_enrichment')) | {'sapiensid.global_matching.enabled'} | INHERITED_SWITCHES


def order_patch(config, body_first):
    """Hold inherited graph/turn settings fixed while comparing formation order."""
    from .body_identity import turn_feature
    settings = config.data['sapiensid']['identities']
    result = {'sapiensid.identities.body_first_matching': body_first}
    for key in ('extended_track_graph', 'turn_recovery'):
        if settings.get(key) is None:
            result['sapiensid.identities.'+key] = turn_feature(settings, key)
    return result


def local_global_variants(config):
    from .body_identity import enabled
    if not enabled(config):
        return
    local = 'sapiensid.identities.'
    views = {'clustering.anchor_view_matching': True} if restricted(config.data['clustering']) else {}
    yield ('local_body', 'Local body paths, global strict face / View',
           {**order_patch(config, True), local+'tracklet_matching': True, **views})
    yield ('local_body', 'Local body paths, isolate global strict face / View',
           {**order_patch(config, True), local+'tracklet_matching': True,
            'sapiensid.global_matching.enabled': False, **views})
    yield ('local_face', 'Local same-scene face support, global body consolidation',
           {**order_patch(config, False), 'sapiensid.global_matching.enabled': True})


def local_global_plans(config, routes):
    """Vary local formation while keeping both global similarity floors fixed."""
    fixed = {key: parameter(config.data, key) for key in (
        'recognition.match_threshold', 'clustering.cohesion_threshold',
        'sapiensid.global_matching.match_threshold', 'sapiensid.global_matching.cohesion_threshold')}
    for family, title, settings in local_global_variants(config):
        for route in routes.get(family, ()):
            patch = {**route, **settings, **fixed}
            yield dict(strategy=family+'_global', title=title, patch=patch,
                       stages=['local_formation', 'global_matching'])


def local_face_routes(config, face_proposals):
    """Translate measured face boundaries into a local bonus when it can help.

The shared cohesion floor is kept intact. A bonus cannot overcome an original
core veto or establish that the measured witnesses belong to the same scene.
    """
    floor = config.data['recognition']['match_threshold']
    cohesion = config.data['clustering']['cohesion_threshold']
    for proposal in face_proposals:
        score = proposal.get('recognition.match_threshold')
        if score is None or not cohesion <= score < floor:
            continue
        bonus = at_or_above(floor-score)
        if 0 < bonus <= .1:
            yield {'clustering.same_scene_match_bonus': max(config.data['clustering']['same_scene_match_bonus'], bonus)}


def view_plans(config, cohorts, followups):
    """Formation and its downstream gates are proposed together, as well as apart."""
    for cohort in cohorts:
        formation = cohort.get('patch')
        if not formation and not cohort.get('supported'):
            continue
        formation = formation or {}
        base = dict(strategy='view_formation', title='Build a supported strict View',
                    patch=formation, stages=['admission', 'strict_core', 'view_support'],
                    witnesses=cohort['witnesses'], cohort=cohort['key'])
        yield base
        for family in ('face', 'hybrid'):
            for attach in followups.get(family, ())[:2]:
                combined = dict(attach)
                for key,value in formation.items():
                    if key in combined and type(value) in (int,float):
                        choose = max if key.endswith(('pose_deviation','match_bonus')) else min
                        combined[key] = choose(combined[key], value)
                    else:
                        combined[key] = value
                yield dict(base, strategy='view_then_'+family,
                           title=('Build a supported View, then match through ' if formation else
                                  'Use a supported View to match through ')+family,
                           patch=combined, stages=base['stages']+['global_'+family])
        if restricted(config.data['clustering']):
            enriched = dict(formation, **{
                'clustering.auxiliary_matching': True, 'clustering.view_prototypes_enabled': True,
                'clustering.context_view_enrichment': True, 'clustering.corroborated_view_enrichment': True})
            yield dict(base, strategy='view_enrichment', title='Build and corroborate alternate Views',
                       patch=enriched, stages=['independent_confirmation','view_support','view_attachment'])


def plans(config, cohorts=(), *, local=None, followups=None, limit=24):
    from .config import digest
    from .merge_solver_search import interleave
    seen = set()
    streams = (view_plans(config, cohorts, followups or {}),
               local_global_plans(config, local or {}))
    for item in interleave(streams):
        changes = {k:v for k,v in item['patch'].items() if v != parameter(config.data, k)}
        signature = digest(changes)
        if not changes or signature in seen:
            continue
        seen.add(signature)
        yield dict(item, patch=changes)
        if len(seen) >= limit:
            return
