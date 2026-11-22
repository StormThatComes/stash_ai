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
        if peer.get('head_error') is not None:
            lines.append('  Rival head/landmark error: '+number(peer['head_error'])+
                         ' (required error advantage '+number(limits.get('rival_margin'))+')')
        if 'own_mask_support' in peer:
            lines.append(f"  Own/peer mask support: {number(peer['own_mask_support'])} / {number(peer['peer_mask_support'])} "
                         f"(minimum {number(limits.get('mask_support'))}); foreign support: {number(peer['foreign_mask_support'])} "
                         f"(maximum {number(limits.get('foreign_mask_support'))}); mask overlap: {number(peer['mask_overlap'])} "
                         f"(maximum {number(limits.get('mask_overlap'))})")
    temporal = check['temporal']
    lines.append(f"Temporal evidence: {temporal['reason']}; similarity >= {number(temporal['threshold'])}; "
                 f"lead >= {number(temporal['margin'])}")
    for neighbour in temporal['neighbours']:
        lines.append(f"Cue {neighbour['cue']}: {neighbour['reason']}; own {number(neighbour.get('own_score'))}; "
                     f"rival {number(neighbour.get('rival_score'))}")
        if neighbour['rejected']:
            lines.append('  Excluded witnesses: '+', '.join(f'{k}={v}' for k,v in neighbour['rejected'].items()))
    return lines
