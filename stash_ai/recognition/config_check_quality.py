"""Admission, mask and portrait rules for the offline recognition policy check."""
FACE_FIELDS = ('min_face_pixels', 'min_sharpness', 'min_visible_face_fraction')
QUERY = 'matching.face.samples.minimum.'
ANCHOR = 'matching.face.anchors.quality.'
REFINE = 'pose.refinement.'
DEDUP = 'pose.deduplication.'


def _face(c):
    # Compare measurements on newly scanned faces. A retained legacy row may
    # still fail a later gate because its measurement is absent.
    has_band = False
    all_paths = []
    for key in FACE_FIELDS:
        scan, query, anchor = 'face.quality.' + key, QUERY + key, ANCHOR + key
        all_paths.extend((scan, query, anchor))
        floor = max(c.value(scan), c.value(query) or 0)
        if c.value(query) is not None:
            c.detail('face-query-' + key, (scan, query),
                     'The effective query minimum on newly scanned faces is the higher of the two floors.',
                     effective_floor=floor)
        if c.value(anchor) is not None:
            has_band |= c.value(anchor) > floor
            c.interval('face-anchor-' + key, (scan, query, anchor), c.value(anchor), floor,
                       'Anchor and query admission can share this quality floor.')
    scan, anchor = 'face.quality.max_pose_deviation', ANCHOR + 'max_pose_deviation'
    all_paths.extend((scan, anchor))
    limit = 1. if c.value(scan) is None else c.value(scan)
    if c.value(anchor) is not None:
        has_band |= c.value(anchor) < limit
        c.interval('face-anchor-pose', (scan, anchor), limit, c.value(anchor),
                   'A lower anchor maximum keeps more off-axis faces eligible as queries only.')
    if c.value('matching.face.auxiliary.enabled'):
        c.add('face-auxiliary-quality-band', all_paths, 'ok' if has_band else 'advice',
              'An auxiliary face needs to pass query admission and miss at least one anchor requirement.',
              'If a distinct auxiliary quality band is desired, separate one anchor and query cutoff '
              'to admit a distinct band of fully measured auxiliary faces.')


def _pose(c):
    collection, anchor = 'body.collection.', 'matching.body.anchors.'
    for key in ('min_person_pixels', 'min_sharpness', 'min_keypoints'):
        first, second = collection + key, anchor + key
        if c.value(second) is not None and c.value(second) > 0:
            c.effective_max('body-anchor-' + key, (first, second), (c.value(first), c.value(second)),
                            'Body anchors must clear both collection and anchor quality.')

    if c.value('pose.estimator.backend') != 'pmpose':
        return
    output_gates = ('pose.estimator.keypoint_score_threshold', 'pose.estimator.pmpose.presence_threshold',
                    'pose.estimator.pmpose.visibility_threshold')
    c.add('pmpose-output-points', (*output_gates, 'body.collection.min_keypoints'),
          'warning' if any(c.value(path) == 1 for path in output_gates) else 'ok',
          'Usable PMPose joints must strictly exceed every active probability cutoff before body quality counts them.',
          'Set each active output cutoff below 1 so the body keypoint minimum can be reached.')
    for prefix, label in ((REFINE, 'SAM first-pass prompt selection'), (DEDUP, 'PMPose duplicate evidence')):
        if c.value(prefix + 'enabled'):
            for key in ('min_keypoint_score', 'min_presence', 'min_visibility'):
                path = prefix + key
                c.add(prefix + key + '-ceiling', (path,), 'warning' if c.value(path) == 1 else 'ok',
                      label + ' requires this probability to be strictly above the effective cutoff.',
                      'Lower the cutoff below 1 to allow measured points.')
    if c.value(DEDUP + 'enabled'):
        from .pose_deduplication import CORE
        count, fraction = DEDUP + 'min_shared_points', DEDUP + 'min_match_fraction'
        least_fraction = c.value(count) / len(CORE)
        c.detail('duplicate-core-agreement', (count, fraction),
                 'The matched-point count already supplies a lower bound on the agreement fraction.',
                 fraction_implied_by_count=least_fraction, effective_minimum=max(least_fraction, c.value(fraction)))
    if c.value(REFINE + 'enabled'):
        base, recovery = REFINE + 'min_mask_score', REFINE + 'min_recovery_mask_score'
        explicit = c.value(recovery)
        floor = (1 + c.value(base)) / 2 if explicit is None else explicit
        c.effective_max('recovery-mask-score', (base, recovery), (floor, c.value(base)),
                        'Sparse masks first pass the ordinary gate. An inherited recovery floor allows an independent-support exception.')
        minimum, maximum = REFINE + 'min_prompt_points', REFINE + 'max_prompt_points'
        if c.value(REFINE + 'adaptive_decoder_retries'):
            c.interval('mask-retry-prompt-count', (minimum, maximum), c.value(maximum), c.value(minimum),
                       'Retries can change prompt selection, negative prompts or mask variants at the same point count.')
        coverage = REFINE + 'min_prompt_coverage'
        n = c.value(maximum)
        c.detail('mask-prompt-coverage', (maximum, coverage),
                 'Coverage above this fraction requires every positive prompt at all allowed counts.',
                 largest_fraction_with_one_miss=(n - 1) / n)
        if c.value(REFINE + 'mode') == 'overlap':
            path = 'pose.detector.max_people_per_frame'
            c.covers('overlap-refinement-people', (REFINE + 'mode', path), c.value(path), 2,
                     'Overlap-only mask refinement needs at least two detected boxes in the frame.',
                     'Allow two detections or use all-person refinement.')


def _portraits(c):
    if not c.value('portraits.enabled') or c.value('portraits.selection.kind') == 'person':
        return
    for key, weight in (('min_face_pixels', 'resolution'), ('min_sharpness', 'sharpness'),
                        ('min_visible_face_fraction', 'visibility'), ('max_pose_deviation', 'frontal'),
                        ('max_clipped_fraction', 'exposure')):
        target, importance = 'portraits.quality.' + key, 'portraits.weights.' + weight
        c.detail('portrait-weight-' + weight, (target, importance),
                 'Zero weight omits this component from the primary portrait score.')
        if c.value('portraits.selection.anchor_only') and key in (*FACE_FIELDS, 'max_pose_deviation') and c.value(importance):
            anchor = ANCHOR + key
            gates = (anchor,) if key == 'max_pose_deviation' else (anchor, QUERY + key)
            values = [c.value(path) for path in gates if c.value(path) is not None]
            if values:
                hard = max(values)
                soft = c.value(target)
                c.detail('portrait-anchor-' + key, (target, *gates),
                         'A soft portrait target adjusts the ranking penalty for missing the target.',
                         admission_limit=hard, target=soft)
    opened, closed = 'portraits.weights.open_mouth', 'portraits.weights.closed_mouth'
    if c.value(opened) and c.value(closed):
        c.detail('portrait-mouth-weights', (opened, closed),
                 'Equal weights give the same combined score to open and closed mouths with measured expression.')


def quality(c):
    _face(c)
    if c.value('body.enabled'):
        _pose(c)
    _portraits(c)
