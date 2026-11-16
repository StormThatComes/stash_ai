"""Threshold relationships in strict, contextual, view and multimodal matching."""
FACE = 'matching.face.'
VIEW = FACE + 'views.'
SCENE = FACE + 'same_scene.'
CROSS = FACE + 'cross_scene.'
BODY = 'matching.body.identities.'
GLOBAL = 'matching.body.global.'
HYBRID = 'matching.hybrid.'


def _mean_and_minimum(c, name, match, cohesion, *, base=None):
    paths = (match, cohesion) if base is None else (match, base, cohesion)
    threshold = max(c.value(match), c.value(base)) if base else c.value(match)
    c.band(name, paths, threshold, c.value(cohesion),
           'Every reference must pass cohesion. The mean cannot fall below that minimum.',
           'Allow a lower cohesion floor if some references may be weaker than the required mean.')


def _rival_headroom(c, route, thresholds, margins):
    # Conditional scenario, since a rival can also score below admission.
    cutoff = max(c.value(path) for path in thresholds)
    margin = max(c.value(path) for path in margins)
    if margin:
        c.band(route + '-rival-headroom', (*thresholds, *margins), 1., cutoff + margin,
               'If a competing identity scores exactly at the match cutoff, the winner needs '
               'that score plus the ambiguity margin. Cosine similarity tops out at 1.',
               'Check labelled rival scores before changing the margin or the match cutoff.',
               collision='advice')


def _views(c):
    match, margin = FACE + 'match_threshold', FACE + 'ambiguity_margin'
    for key, ordinary in (('match_threshold', match), ('ambiguity_margin', margin)):
        setting = VIEW + key
        c.band('view-' + key, (ordinary, setting), c.value(setting), c.value(ordinary),
               'View-only matching uses the larger of these two values.',
               'Raise the View setting if this route should require stronger evidence.')
    _mean_and_minimum(c, 'view-mean-cohesion', VIEW + 'match_threshold',
                      VIEW + 'cohesion_threshold', base=match)
    _rival_headroom(c, 'view', (match, VIEW + 'match_threshold'), (margin, VIEW + 'ambiguity_margin'))
    reflection = CROSS + 'cooccurrence_merge_threshold'
    c.band('view-reflection', (reflection, VIEW + 'match_threshold', match, VIEW + 'cohesion_threshold'),
           c.value(reflection), max(c.value(match), c.value(VIEW + 'match_threshold'),
                                    c.value(VIEW + 'cohesion_threshold')),
           'For co-present View merges the reflection floor can add a stronger per-pair test.',
           'Raise the reflection floor if co-present faces require extra similarity.')

    support = VIEW + 'min_support'
    routes = (
        (VIEW + 'anchor_view_matching', 'local-anchor-view', SCENE + 'min_support'),
        (VIEW + 'strict_appearance_matching', 'historical-strict-appearance', CROSS + 'min_merge_support'),
        (VIEW + 'prototypes_enabled', 'verified-auxiliary-view', FACE + 'auxiliary.min_support'),
    )
    for flag, route, other in routes:
        if c.value(flag):
            c.covers(route + '-support', (support, other), c.value(support), c.value(other),
                     f'{route} uses the larger support count, {max(c.value(support), c.value(other))}.',
                     'To lower support on this route, review the setting that supplies the larger count.')
    if c.value(VIEW + 'anchor_view_matching'):
        attach = VIEW + 'attach_min_support'
        c.covers('pose-recovery-view-support', (attach,), c.value(attach), 2,
                 'Uncertain-pose recovery requires at least two query witnesses even when View attachment needs one.',
                 'Use ordinary View attachment for single-witness claims or review the recovery support target.')
        other = SCENE + 'min_support'
        c.scene_capacity('local-view-cue-cap', (support, other), max(c.value(support), c.value(other)),
                         'Building a strict anchor View')
    if c.value(VIEW + 'context_view_enrichment'):
        bonus = SCENE + 'match_bonus'
        c.add('context-view-source', (VIEW + 'context_view_enrichment', bonus),
              'warning' if c.value(bonus) == 0 else 'ok',
              'Context View enrichment consumes same-scene attachments produced by a positive face bonus.',
              'Use a positive face bonus to enable that source or turn off contextual enrichment.')
    if c.value(VIEW + 'prototypes_enabled'):
        support, cap = VIEW + 'attach_min_support', FACE + 'samples.references.max_samples_per_person'
        c.covers('early-view-core-repair', (support, cap), c.value(cap), c.value(support),
                 'Early auxiliary View repair needs this many selected strict samples from the core.',
                 'Increase the face reference bank or reduce View attachment support for this repair.')


def _body(c):
    match, cohesion = BODY + 'match_threshold', BODY + 'cohesion_threshold'
    _mean_and_minimum(c, 'local-body-mean-cohesion', match, cohesion)
    for key, label in (('min_support', 'Confirming a local body core'),
                       ('attach_min_support', 'Building an independent face-linked body bank')):
        c.scene_capacity('local-body-cue-cap-' + key, (BODY + key,), c.value(BODY + key), label)
    _rival_headroom(c, 'local-body', (match,), (BODY + 'ambiguity_margin',))
    track_path = BODY + 'track_similarity_threshold'
    track = c.value(track_path)
    track_paths = (track_path,)
    if track is None:
        track = c.value(cohesion)
        track_paths += (cohesion,)
    # Perspective recovery also checks transitions with tracklets disabled.
    c.band('body-adjacent-cues', (*track_paths, match), c.value(match), track,
           'A visually continuous adjacent-cue edge compares one pair. Null inherits local cohesion.',
           'Lower the track threshold if continuity should admit weaker pairs than ordinary local matching.',
           collision='advice')
    margin = BODY + 'ambiguity_margin'
    c.covers('body-track-margin', (margin,), c.value(margin), .025,
             'Temporal transitions enforce a minimum rival lead of 0.025.',
             'Values below 0.025 affect other local routes only.')
    flags = (BODY + 'tracklet_matching', BODY + 'body_first_matching')
    if c.value(flags[1]) or c.value(BODY + 'extended_track_graph') or c.value(BODY + 'turn_recovery'):
        c.add('body-track-features', (*flags, BODY + 'extended_track_graph', BODY + 'turn_recovery'),
              'ok' if c.value(flags[0]) else 'warning',
              'Body-first graph construction, extended tracks and turn recovery require tracklet matching.',
              'Enable tracklets to use the requested graph features.')
    from .body_identity import turn_feature
    if turn_feature(c.config.data['sapiensid']['identities'], 'turn_recovery'):
        discount = BODY + 'turn_similarity_discount'
        floor = max(0., track - c.value(discount))
        c.band('body-turn-discount', (*track_paths, discount), track, floor,
               'At a measured turn, the transition floor is the track threshold minus the discount, bounded at zero.',
               'Increase the discount if verified turns should have a wider recovery interval.')
        c.covers('body-turn-discount-cap', (*track_paths, discount), track, c.value(discount),
                 'Any part of the turn discount above the track threshold is clipped at zero.',
                 'Review the turn discount against the effective track threshold.')

    bonus_path = BODY + 'scene_match_bonus'
    bonus_paths = (bonus_path,)
    bonus = c.value(bonus_path)
    if bonus is None:
        bonus_paths += (SCENE + 'match_bonus',)
        bonus = c.value(SCENE + 'match_bonus')
    if bonus:
        high, low = c.value(match), c.value(cohesion)
        c.band('body-scene-bonus', (match, cohesion, *bonus_paths), high, max(high - bonus, low),
               'Same-scene roster attachment discounts the mean. Each witness still has to pass cohesion. '
               'The interval shown assumes full scene weight.',
               'Review local cohesion and the scene bonus together to leave room below the ordinary match.')
        required = (BODY + 'attach_min_support', SCENE + 'min_support')
        c.scene_capacity('body-roster-cue-cap', required, 1 + max(2, *(c.value(p) for p in required)),
                         'A roster query and its independent witnesses in other shots')
    if c.value(GLOBAL + 'enabled'):
        _mean_and_minimum(c, 'global-body-mean-cohesion', GLOBAL + 'match_threshold', GLOBAL + 'cohesion_threshold')
        _rival_headroom(c, 'global-body', (GLOBAL + 'match_threshold',), (GLOBAL + 'ambiguity_margin',))
        for key in ('match_threshold', 'cohesion_threshold'):
            local, remote = BODY + key, GLOBAL + key
            c.band('global-local-body-' + key, (local, remote), c.value(remote), c.value(local),
                   'Across-scene and local body matching use these cutoffs on their respective reference banks.',
                   'Consider a stricter global cutoff if cross-scene merges need stronger similarity.',
                   collision='advice')
        local, remote = BODY + 'min_support', GLOBAL + 'min_support'
        c.covers('provisional-global-body-support', (remote, local), c.value(remote), c.value(local),
                 'A provisional global merge needs the larger combined support count across at least two scans.',
                 'Review both support settings when tuning provisional global merges.')


def matching(c):
    match, cohesion = FACE + 'match_threshold', FACE + 'cohesion_threshold'
    _mean_and_minimum(c, 'face-mean-cohesion', match, cohesion)
    _rival_headroom(c, 'face', (match,), (FACE + 'ambiguity_margin',))
    reflection = CROSS + 'cooccurrence_merge_threshold'
    c.band('face-reflection', (reflection, match), c.value(reflection), c.value(match),
           'Co-present face merges use the larger threshold on the weakest cross-pair.',
           'Raise the reflection threshold to require extra similarity in the same cue.')
    support = FACE + 'anchors.min_identity_support'
    c.scene_capacity('local-face-birth-cue-cap', (support,), c.value(support), 'Confirming a new face core locally')
    if c.value(CROSS + 'provisional'):
        c.covers('provisional-face-birth', (CROSS + 'provisional', support), c.value(support), 2,
                 'Cross-scene pooling uses groups withheld for insufficient local support. '
                 'At one vote a new singleton can already confirm locally.',
                 'Increase birth support if confirmation should depend on independent evidence.')
    bonus = SCENE + 'match_bonus'
    if c.value(bonus):
        for setting in (match, cohesion):
            c.band('face-same-scene-' + setting.split('.')[-1], (setting, bonus), c.value(setting),
                   max(0., c.value(setting) - c.value(bonus)),
                   'Strict local support reduces this face cutoff by the scene bonus, down to zero.',
                   'Review the bonus if borderline same-scene faces need a wider interval.')
        support = SCENE + 'min_support'
        c.scene_capacity('face-context-cue-cap', (support,), c.value(support), 'Establishing a strict same-scene core')
    if c.value(VIEW + 'anchor_view_matching') or c.value(VIEW + 'prototypes_enabled'):
        _views(c)
    if c.value('body.enabled'):
        _rival_headroom(c, 'hybrid-body', (HYBRID + 'person_threshold',), (HYBRID + 'ambiguity_margin',))
        cap, required = FACE + 'samples.references.max_samples_per_person', HYBRID + 'min_support'
        c.covers('hybrid-selected-bank-support', (cap, required), c.value(cap), c.value(required),
                 'Without a supported body mode, hybrid support is counted from the selected reference bank. '
                 'Supported modes can count independent witnesses beyond that bank.',
                 'Increase the bank size or review hybrid support for the selected-bank fallback.')
        floor = HYBRID + 'face_floor'
        for path, label in ((cohesion, 'weakest face reference'), (match, 'ordinary face mean')):
            c.band('hybrid-face-' + path.split('.')[-1], (floor, path), c.value(path), c.value(floor),
                   f'Hybrid body-assisted matching first requires every face reference to pass the face floor. '
                   f'This is the room below the {label} cutoff.',
                   'Lower the hybrid face floor if body evidence should recover weaker face matches. '
                   'This floor also controls face contradictions for global body merges.')
        if c.value(BODY + 'enabled'):
            _body(c)
