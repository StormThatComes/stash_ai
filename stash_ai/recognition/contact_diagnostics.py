"""Readable recorded contact checks for assigned and withheld scene details."""


def describe(check):
    if not check:
        return ['Detailed contact evidence was not retained. Run recognition again to collect it.']
    def number(value):
        return 'unavailable' if value is None else f'{value:.4f}'
    lines = [f"Contact face check: {check['status'].upper()} · {check['reason']}",
             'Recorded before subsequent track recovery; final membership is shown above.']
    crowd = check['crowding']
    lines.append(f"Crowding: {crowd['detected_people']} people; overlap {number(crowd['overlap'])}; "
                 f"box overlap {number(crowd['box_overlap'])}; trigger {number(crowd['threshold'])}")
    local = check['same_frame']
    limits = local.get('thresholds', {})
    lines.append('Same-cue face/body evidence: '+local['reason'])
    if local.get('disabled_mask_settings'):
        lines.append('Own contact mask disabled by: '+', '.join(local['disabled_mask_settings']))
    if 'full_body_reason' in local:
        lines.append('Full head/torso path: '+local['full_body_reason'])
    if 'head_points' in local:
        names = ('nose','left_eye','right_eye')
        lines.append('Independent pose head points: '+(', '.join(names[i] for i in local['head_points']) or 'none'))
        lines.append('Maximum head/landmark distance / face side: '+number(local.get('head_error'))+
                     ' (limit '+number(limits.get('head_error'))+')')
    if 'torso_points' in local:
        lines.append(f"Independent torso points: {len(local['torso_points'])}; need both shoulders and at least one hip")
    if 'head_projection' in local:
        lines.append(f"Head/torso projection: {number(local['head_projection'])} "
                     f"(range {number(limits.get('head_projection_min'))}..{number(limits.get('head_projection_max'))}); "
                     f"lateral offset: {number(local['head_lateral'])} (limit {number(local['head_lateral_limit'])})")
    for peer in local.get('peers', ()):
        lines.append(f"Peer {peer.get('observation', peer['person_index'])}: {peer['reason']}")
        if peer.get('disabled_mask_settings'):
            lines.append('  Peer contact mask disabled by: '+', '.join(peer['disabled_mask_settings']))
        if peer.get('head_error') is not None:
            lines.append('  Rival head/landmark error: '+number(peer['head_error'])+
                         ' (required error advantage '+number(limits.get('rival_margin'))+')')
        if 'own_mask_support' in peer:
            lines.append(f"  Own/peer mask support: {number(peer['own_mask_support'])} / {number(peer['peer_mask_support'])} "
                         f"(minimum {number(limits.get('mask_support'))}); foreign support: {number(peer['foreign_mask_support'])} "
                         f"(maximum {number(limits.get('foreign_mask_support'))}); mask overlap: {number(peer['mask_overlap'])} "
                         f"(maximum {number(limits.get('mask_overlap'))})")
    upper = local.get('upper_body')
    if upper:
        names = ('nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder')
        def points(indices):
            return ', '.join(names[i] for i in indices) or 'none'
        lines.append('Masked upper-body path: '+upper['reason'])
        lines.append(f"  Original aligned head points: {points(upper['head_points'])}; "
                     f"supporting ears: {points(upper['ear_points'])}; shoulders: {points(upper['shoulder_points'])}")
        lines.append('  Checks: '+', '.join(f'{k}={v}' for k,v in upper['checks'].items()))
        if upper.get('excluded_points'):
            lines.append('  Excluded original points: '+', '.join(
                f"{names[p['point']]}={p['reason']}" for p in upper['excluded_points']))
        for ear in upper.get('ears', ()):
            lines.append(f"  {names[ear['point']]} distance / face side: {number(ear['distance'])}; "
                         f"anatomically eligible: {ear['eligible']}")
        if upper['checks']['head'] == 'insufficient_head_points':
            lines.append('  Need two original nose/eye points, or one original eye with its same-side original ear. '
                         'Missing, borrowed, occluded and out-of-box points cannot support this path.')
        if upper.get('shoulder_mode') == 'single':
            lines.append('  Single-shoulder path: original head support, all five face landmarks in the mask, '
                         'and a masked head-to-shoulder connection; no shoulder is synthesized.')
            if 'neck_distance' in upper:
                single_limits = upper['thresholds']
                lines.append(f"  Head/shoulder distance / face side: {number(upper['neck_distance'])}; "
                             f"projection along face axis: {number(upper['head_projection'])}; "
                             f"lateral: {number(upper['head_lateral'])}")
                lines.append(f"  Single-shoulder limits: distance "
                             f"{number(single_limits['single_shoulder_distance_min'])}..{number(single_limits['single_shoulder_distance_max'])}; "
                             f"projection {number(single_limits['single_shoulder_projection_min'])}..{number(single_limits['single_shoulder_projection_max'])}; "
                             f"lateral <= {number(single_limits['single_shoulder_lateral_max'])}")
            if 'face_landmark_mask_support' in upper:
                lines.append('  Face landmarks inside own mask: '+number(upper['face_landmark_mask_support'])+' (required 1.0000)')
        elif 'neck_distance' in upper:
            lines.append(f"  Head/shoulder distance: {number(upper['neck_distance'])}; "
                         f"lateral: {number(upper['head_lateral'])}; normal: {number(upper['neck_normal'])}; hips optional")
        if upper.get('mask'):
            mask = upper['mask']
            lines.append(f"  Own mask anatomy/corridor support: {number(mask['anatomy_support'])} / "
                         f"{number(mask['corridor_support'])}; longest gap / face side: {number(mask['max_gap'])} "
                         f"(support >= {number(upper['thresholds']['mask_support'])}; "
                         f"gap <= {number(upper['thresholds']['mask_gap'])})")
        for peer in upper['peers']:
            lines.append(f"  Upper-body peer {peer.get('observation', peer['person_index'])}: {peer['reason']}")
            if peer.get('disabled_mask_settings'):
                lines.append('    Peer contact mask disabled by: '+', '.join(peer['disabled_mask_settings']))
            if 'local_box_separated' in peer:
                lines.append(f"    Head/shoulder region outside peer box: {peer['local_box_separated']}")
            if 'foreign_mask_support' in peer:
                lines.append(f"    Foreign upper-body/head support: {number(peer['foreign_mask_support'])} / "
                             f"{number(peer['foreign_head_support'])}")
    temporal = check['temporal']
    lines.append(f"Temporal evidence: {temporal['reason']}; similarity >= {number(temporal['threshold'])}; "
                 f"lead >= {number(temporal['margin'])}")
    for neighbour in temporal['neighbours']:
        lines.append(f"Cue {neighbour['cue']}: {neighbour['reason']}; own {number(neighbour.get('own_score'))}; "
                     f"rival {number(neighbour.get('rival_score'))}")
        if neighbour['rejected']:
            lines.append('  Excluded witnesses: '+', '.join(f'{k}={v}' for k,v in neighbour['rejected'].items()))
    return lines
