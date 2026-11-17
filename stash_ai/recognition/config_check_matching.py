"""Effective matching thresholds and the routes available under those thresholds."""
from math import isclose

FACE = 'matching.face.'
VIEW = FACE + 'views.'
SCENE = FACE + 'same_scene.'
CROSS = FACE + 'cross_scene.'
BODY = 'matching.body.identities.'
GLOBAL = 'matching.body.global.'
HYBRID = 'matching.hybrid.'


def _face_routes(c):
    support = FACE + 'anchors.min_identity_support'
    c.scene_capacity('local-face-birth-cue-cap', (support,), c.value(support), 'Confirming a new face core locally')
    provisional = CROSS + 'provisional'
    c.route('provisional-face-birth', (provisional, support), c.value(support) >= 2,
            'Cross-scene pooling uses groups withheld for insufficient local support. '
            'At one vote a singleton can already confirm locally.',
            'Review birth support if cross-scene provisional pooling is needed.',
            enabled=c.value(provisional), available=c.value(support), required=2)
    bonus, local = SCENE + 'match_bonus', SCENE + 'min_support'
    context = VIEW + 'context_view_enrichment'
    c.route('context-view-source', (context, bonus), c.value(bonus) > 0,
            'Context View enrichment consumes same-scene attachments produced by a positive face bonus.',
            'Use a positive face bonus if this source is needed.', enabled=c.value(context))
    if c.value(bonus):
        c.scene_capacity('face-context-cue-cap', (local,), c.value(local), 'Establishing a strict same-scene core')
    if c.value(VIEW + 'anchor_view_matching'):
        support = VIEW + 'min_support'
        c.scene_capacity('local-view-cue-cap', (support, local), max(c.value(support), c.value(local)),
                         'Building a strict anchor View')
    cap = FACE + 'samples.references.max_samples_per_person'
    if c.value(VIEW + 'prototypes_enabled'):
        support = VIEW + 'attach_min_support'
        c.route('early-view-core-repair', (support, cap), c.value(cap) >= c.value(support),
                'Early auxiliary View repair counts selected strict samples from the core.',
                'Increase the face reference bank or reduce View attachment support for this repair.',
                available=c.value(cap), required=c.value(support))
    if c.value('body.enabled'):
        support = HYBRID + 'min_support'
        c.route('hybrid-selected-bank-support', (cap, support), c.value(cap) >= c.value(support),
                'Without a supported body mode, hybrid support is counted from the selected reference bank. '
                'Supported modes can count independent witnesses beyond that bank.',
                'Increase the bank size or review hybrid support for the selected-bank fallback.',
                available=c.value(cap), required=c.value(support))


def _body_routes(c, match, cohesion):
    for key, label in (('min_support', 'Confirming a local body core'),
                       ('attach_min_support', 'Building an independent face-linked body bank')):
        c.scene_capacity('local-body-cue-cap-' + key, (BODY + key,), c.value(BODY + key), label)
    bonus_path = BODY + 'scene_match_bonus'
    paths = (match, cohesion, bonus_path)
    bonus = c.value(bonus_path)
    if bonus is None:
        inherited = SCENE + 'match_bonus'
        paths += (inherited,)
        bonus = c.value(inherited)
    high = c.value(match)
    low = max(high - bonus, c.value(cohesion))
    c.route('body-scene-bonus', paths, low < high,
            'The roster bonus discounts the mean. Cohesion still applies to each witness.',
            'Keep local body cohesion below the match cutoff to let the bonus admit a weaker mean.',
            enabled=bonus > 0, lower=low, upper=high, gap=max(0., high-low))
    if bonus:
        required = (BODY + 'attach_min_support', SCENE + 'min_support')
        c.scene_capacity('body-roster-cue-cap', required, 1 + max(2, *(c.value(p) for p in required)),
                         'A roster query and its independent witnesses in other shots')


def solver_constraints(c):
    """Recognition policy enforced before replay. Keep scan and ranking diagnostics separate."""
    from .body_identity import enabled
    _face_routes(c)
    if not enabled(c.config):
        return
    match, track, cohesion = (BODY + key for key in ('match_threshold', 'track_similarity_threshold', 'cohesion_threshold'))
    _body_routes(c, match, cohesion)
    lower = c.value(track)
    paths = (track, match)
    if lower is None:
        lower = c.value(cohesion)
        paths = (track, cohesion, match)
    upper = c.value(match)
    gap = upper - lower
    wide = gap > 0 and (gap >= c.score_gap or isclose(gap, c.score_gap, rel_tol=1e-12))
    c.add('body-adjacent-cues', paths, 'ok' if wide else 'advice',
          'A visually continuous adjacent-cue edge compares one pair. Null inherits local cohesion.',
          'Leave room below the ordinary local match threshold for adjacent-cue continuity.',
          solver_constraint=True, lower=lower, upper=upper, gap=gap, minimum_gap=c.score_gap)


def _mean_and_minimum(c, name, match, cohesion, *, base=None):
    paths = (match, cohesion) if base is None else (match, base, cohesion)
    threshold = max(c.value(match), c.value(base)) if base else c.value(match)
    c.interval(name, paths, threshold, c.value(cohesion),
               'The weakest reference must pass cohesion and the mean must pass the match cutoff.')


def _rival_headroom(c, route, thresholds, margins):
    cutoff = max(c.value(path) for path in thresholds)
    margin = max(c.value(path) for path in margins)
    if margin:
        c.detail(route + '-rival-headroom', (*thresholds, *margins),
                 'Required winning score for a hypothetical rival exactly at the cutoff. Actual rival scores vary.',
                 rival_at_cutoff=cutoff, required_winner=cutoff + margin)


def _views(c):
    match, margin = FACE + 'match_threshold', FACE + 'ambiguity_margin'
    for key, ordinary in (('match_threshold', match), ('ambiguity_margin', margin)):
        setting = VIEW + key
        c.effective_max('view-' + key, (ordinary, setting), (c.value(ordinary), c.value(setting)),
                        'View matching uses the higher ordinary or View cutoff.')
    _mean_and_minimum(c, 'view-mean-cohesion', VIEW + 'match_threshold',
                      VIEW + 'cohesion_threshold', base=match)
    _rival_headroom(c, 'view', (match, VIEW + 'match_threshold'), (margin, VIEW + 'ambiguity_margin'))
    paths = (CROSS + 'cooccurrence_merge_threshold', VIEW + 'match_threshold', match, VIEW + 'cohesion_threshold')
    c.effective_max('view-reflection', paths, map(c.value, paths),
                    'Co-present View merges apply the effective floor to the weakest cross-pair.')

    support = VIEW + 'min_support'
    routes = (
        (VIEW + 'anchor_view_matching', 'local-anchor-view', SCENE + 'min_support'),
        (VIEW + 'strict_appearance_matching', 'historical-strict-appearance', CROSS + 'min_merge_support'),
        (VIEW + 'prototypes_enabled', 'verified-auxiliary-view', FACE + 'auxiliary.min_support'),
    )
    for flag, route, other in routes:
        if c.value(flag):
            c.effective_max(route + '-support', (support, other), (c.value(support), c.value(other)),
                            f'{route} uses the larger support count.')
    if c.value(VIEW + 'anchor_view_matching'):
        attach = VIEW + 'attach_min_support'
        c.effective_max('pose-recovery-view-support', (attach,), (c.value(attach), 2),
                        'Uncertain-pose recovery needs at least two query witnesses.')


def _body(c):
    match, cohesion = BODY + 'match_threshold', BODY + 'cohesion_threshold'
    _mean_and_minimum(c, 'local-body-mean-cohesion', match, cohesion)
    _rival_headroom(c, 'local-body', (match,), (BODY + 'ambiguity_margin',))
    track_path = BODY + 'track_similarity_threshold'
    track = c.value(track_path)
    track_paths = (track_path,)
    if track is None:
        track = c.value(cohesion)
        track_paths += (cohesion,)
    margin = BODY + 'ambiguity_margin'
    c.effective_max('body-track-margin', (margin,), (c.value(margin), .025),
                    'Temporal transitions enforce a minimum rival lead of 0.025.')
    flags = tuple(BODY + key for key in ('tracklet_matching', 'body_first_matching', 'extended_track_graph', 'turn_recovery'))
    if any(c.value(key) for key in flags[1:]):
        c.add('body-track-features', flags, 'ok' if c.value(flags[0]) else 'warning',
              'The tracklet graph, extended track edges and turn recovery require tracklets. '
              'Body-first seed ordering can still operate independently.',
              'Enable tracklets if these graph stages are needed.')
    from .body_identity import turn_feature
    if turn_feature(c.config.data['sapiensid']['identities'], 'turn_recovery'):
        discount = BODY + 'turn_similarity_discount'
        floor = max(0., track - c.value(discount))
        c.interval('body-turn-discount', (*track_paths, discount), track, floor,
                   'Measured turns use the discounted track threshold, bounded at zero.')
        c.detail('body-turn-discount-cap', (*track_paths, discount),
                 'The effective discount ends at a zero similarity floor.', effective=track-floor)

    if c.value(GLOBAL + 'enabled'):
        _mean_and_minimum(c, 'global-body-mean-cohesion', GLOBAL + 'match_threshold', GLOBAL + 'cohesion_threshold')
        _rival_headroom(c, 'global-body', (GLOBAL + 'match_threshold',), (GLOBAL + 'ambiguity_margin',))
        for key in ('match_threshold', 'cohesion_threshold'):
            local, remote = BODY + key, GLOBAL + key
            c.interval('global-local-body-' + key, (local, remote), c.value(remote), c.value(local),
                       'Across-scene and local body matching score different reference banks.')
        local, remote = BODY + 'min_support', GLOBAL + 'min_support'
        c.effective_max('provisional-global-body-support', (remote, local), (c.value(remote), c.value(local)),
                        'A provisional global merge needs the larger combined support across at least two scans.')


def matching(c):
    match, cohesion = FACE + 'match_threshold', FACE + 'cohesion_threshold'
    _mean_and_minimum(c, 'face-mean-cohesion', match, cohesion)
    margin = FACE + 'ambiguity_margin'
    c.add('face-ambiguity-margin', (margin,), 'ok' if c.value(margin) > 0 else 'advice',
          'A zero margin allows tied face scores to pass the ordinary rival test.',
          'Set a positive margin to require a lead over a competing face identity.')
    _rival_headroom(c, 'face', (match,), (margin,))
    reflection = CROSS + 'cooccurrence_merge_threshold'
    c.add('face-reflection', (reflection, match), 'ok' if c.value(reflection) > c.value(match) else 'advice',
          'Co-present face merges test the weakest cross-pair against the higher cutoff.',
          'Raise the co-occurrence threshold above the ordinary match cutoff for extra separation.',
          effective=max(c.value(reflection), c.value(match)))
    bonus = SCENE + 'match_bonus'
    if c.value(bonus):
        for setting in (match, cohesion):
            c.interval('face-same-scene-' + setting.split('.')[-1], (setting, bonus), c.value(setting),
                       max(0., c.value(setting) - c.value(bonus)), 'Local face support discounts this cutoff down to zero.')
    if c.value(VIEW + 'anchor_view_matching') or c.value(VIEW + 'prototypes_enabled'):
        _views(c)
    if c.value('body.enabled'):
        _rival_headroom(c, 'hybrid-body', (HYBRID + 'person_threshold',), (HYBRID + 'ambiguity_margin',))
        floor = HYBRID + 'face_floor'
        for path, label in ((cohesion, 'weakest face reference'), (match, 'ordinary face mean')):
            c.interval('hybrid-face-' + path.split('.')[-1], (floor, path), c.value(path), c.value(floor),
                       f'Room below the {label} cutoff for body-assisted matching. The face floor also gates contradictions.')
        if c.value(BODY + 'enabled'):
            _body(c)
