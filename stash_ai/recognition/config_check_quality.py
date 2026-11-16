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
            c.add('face-query-' + key, (scan, query),
                  'advice' if c.value(query) == c.value(scan) and floor > 0 else 'ok',
                  'The effective query minimum on newly scanned faces is the higher of the two floors.',
                  'Raise the query minimum only if queries should be filtered further after collection.',
                  effective_floor=floor)
        if c.value(anchor) is not None:
            has_band |= c.value(anchor) > floor
            c.band('face-anchor-' + key, (scan, query, anchor), c.value(anchor), floor,
                   'The anchor cutoff can separate anchor and auxiliary faces on this measurement.',
                   'Raise the anchor floor if this dimension should distinguish the roles.',
                   minimum=c.score_gap if key == 'min_visible_face_fraction' else 0, collision='advice')
    scan, anchor = 'face.quality.max_pose_deviation', ANCHOR + 'max_pose_deviation'
    all_paths.extend((scan, anchor))
    limit = 1. if c.value(scan) is None else c.value(scan)
    if c.value(anchor) is not None:
        has_band |= c.value(anchor) < limit
        c.band('face-anchor-pose', (scan, anchor), limit, c.value(anchor),
               'Lower maximum pose deviation makes anchor admission stricter than scan admission.',
               'Lower the anchor maximum if off-axis faces should be auxiliary.', collision='advice')
    if c.value('matching.face.auxiliary.enabled'):
        c.add('face-auxiliary-quality-band', all_paths, 'ok' if has_band else 'advice',
              'An auxiliary face needs to pass query admission and miss at least one anchor requirement.',
              'Separate an anchor and query cutoff on visibility, pose, size or sharpness '
              'to admit a distinct band of fully measured auxiliary faces.')


def _pose(c):
    collection, anchor = 'body.collection.', 'matching.body.anchors.'
    for key in ('min_person_pixels', 'min_sharpness', 'min_keypoints'):
        first, second = collection + key, anchor + key
        if c.value(second) is not None and c.value(second) > 0:
            c.band('body-anchor-' + key, (first, second), c.value(second), c.value(first),
                   'Body/hybrid anchors must clear collection quality as well as anchor quality.',
                   'Raise the anchor requirement if this measurement should distinguish body references.',
                   minimum=0, collision='advice')

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
        c.add('duplicate-core-agreement', (count, fraction), 'advice' if c.value(fraction) <= least_fraction else 'ok',
              'The minimum matched-point count already implies this fraction even with all core points shared.',
              'Raise the fraction if it should filter pairs beyond the point count.',
              fraction_implied_by_count=least_fraction)
    if c.value(REFINE + 'enabled'):
        base, recovery = REFINE + 'min_mask_score', REFINE + 'min_recovery_mask_score'
        explicit = c.value(recovery)
        floor = (1 + c.value(base)) / 2 if explicit is None else explicit
        c.band('recovery-mask-score', (base, recovery), floor, c.value(base),
               'A weak or sparse mask is tested after the ordinary mask gate. Null uses the midpoint '
               'between the base score and 1, with an independent-support exception.',
               'Raise the recovery score if weak masks should need stronger evidence.')
        minimum, maximum = REFINE + 'min_prompt_points', REFINE + 'max_prompt_points'
        if c.value(REFINE + 'adaptive_decoder_retries'):
            c.band('mask-retry-prompt-count', (minimum, maximum), c.value(maximum), c.value(minimum),
                   'Retries starting at the ordinary prompt minimum can add points up to the maximum.',
                   'Increase the maximum if retries should be able to add another positive point.',
                   minimum=0, collision='advice')
        coverage = REFINE + 'min_prompt_coverage'
        n = c.value(maximum)
        threshold = c.value(coverage)
        c.add('mask-prompt-coverage', (maximum, coverage),
              'advice' if (n - 1) / n < threshold < 1 else 'ok',
              'Above this fraction even a single missed positive prompt fails at every allowed prompt count.',
              'Use a reachable fraction if a mask may miss a positive prompt.',
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
        c.add('portrait-weight-' + weight, (target, importance), 'advice' if c.value(importance) == 0 else 'ok',
              'The component weight controls the contribution of this target to the primary face portrait score.',
              'Use a positive weight if the quality target should influence the primary ranking.')
        if c.value('portraits.selection.anchor_only') and key in (*FACE_FIELDS, 'max_pose_deviation') and c.value(importance):
            anchor = ANCHOR + key
            gates = (anchor,) if key == 'max_pose_deviation' else (anchor, QUERY + key)
            values = [c.value(path) for path in gates if c.value(path) is not None]
            if values:
                hard = max(values)
                soft = c.value(target)
                gap = hard - soft if key == 'max_pose_deviation' else soft - hard
                c.add('portrait-anchor-' + key, (target, *gates), 'advice' if gap <= 0 else 'ok',
                      'The soft target adds a shortfall penalty only if an admitted face can miss it.',
                      'Adjust the portrait target if it should further rank anchor-qualified faces.',
                      admission_limit=hard, target=soft)
    opened, closed = 'portraits.weights.open_mouth', 'portraits.weights.closed_mouth'
    if c.value(opened) and c.value(closed):
        c.add('portrait-mouth-weights', (opened, closed), 'advice' if c.value(opened) == c.value(closed) else 'ok',
              'Open and closed mouth scores are complements. Equal weights cancel the mouth preference '
              'on faces with expression measurements.',
              'Give the preferred mouth state the larger weight.')


def quality(c):
    _face(c)
    if c.value('body.enabled'):
        _pose(c)
    _portraits(c)
