"""Reject blur thresholds whose active pipeline cannot produce semantic masks."""

ALGORITHM = 'foreground-directional-reblur-v1'
INPUT_SPACE = 'encoder-input-sharpness-v1'


def validate(data):
    from .pose_policy import anchor_visibility

    detection, grouping, portrait = (data[k] for k in ('detection', 'clustering', 'portraits'))
    face_floors = {
        'detection.min_sharpness': detection['min_sharpness'],
        'clustering.min_auxiliary_sharpness': grouping['min_auxiliary_sharpness'],
        'clustering.min_anchor_sharpness': grouping['min_anchor_sharpness'],
    }
    from .sharpness_gates import policy
    active = [path for path, value in face_floors.items() if value and value > 0]
    def split_active(settings):
        return settings.get('sharpness_mode') == 'split' and any(settings.get(k) for k in ('min_focus_sharpness','min_motion_sharpness'))
    if split_active(detection):active.append('detection.min_focus_sharpness')
    for role in ('anchor', 'auxiliary'):
        if split_active(policy(grouping, role+'_', 'min_'+role+'_sharpness')):
            active.append('clustering.'+role+'_min_focus_sharpness')
    role_visibility = anchor_visibility(grouping)
    if active and not (detection['min_visible_face_fraction'] > 0 or role_visibility > 0):
        raise ValueError(', '.join(active) + ' > 0 requires FaceOcc masks: set '
                         'detection.min_visible_face_fraction, clustering.min_auxiliary_visible_face_fraction '
                         'or clustering.min_anchor_visible_face_fraction > 0, or disable these sharpness floors.')

    portrait_floor = (portrait['weights']['sharpness'] > 0 and portrait['min_sharpness'] > 0)
    anchor_floor = portrait['anchor_only'] and any(grouping[k] and grouping[k] > 0
        for k in ('min_auxiliary_sharpness', 'min_anchor_sharpness'))
    if (portrait['enabled'] and portrait['kind'] != 'person' and (portrait_floor or anchor_floor)
            and not (portrait['weights']['visibility'] > 0 or portrait['anchor_only'] and role_visibility > 0)):
        raise ValueError('Portrait sharpness requires FaceOcc masks: set portraits.weights.visibility > 0 '
                         'or disable portraits.min_sharpness and any portrait anchor sharpness floors.')

    body = data['sapiensid']
    if not body['enabled']:
        return
    # Face-associated hybrid evidence also uses these gates when independent
    # collection/local identities are disabled.
    active = [path for path, value in (
        ('sapiensid.collection.min_sharpness', body['collection']['min_sharpness']),
        ('sapiensid.identities.anchor_min_sharpness', body['identities']['anchor_min_sharpness']),
    ) if value > 0]
    if split_active(body['collection']):active.append('sapiensid.collection.min_focus_sharpness')
    if split_active(policy(body['identities'], 'anchor_', 'anchor_min_sharpness')):
        active.append('sapiensid.identities.anchor_min_focus_sharpness')
    pose, refinement = body['pose'], body['pose']['refinement']
    if active and refinement.get('mask_sharpness', True) and not (pose['backend'] in ('pmpose','vitpose_plus_large','sapiens','rtmo') and refinement['enabled'] and refinement['mode'] == 'all'):
        raise ValueError(', '.join(active) + ' > 0 requires SAM masks for every person: set '
                         'a supported COCO17 pose backend, sapiensid.pose.refinement.enabled: true and '
                         'sapiensid.pose.refinement.mode: all, or set both sharpness floors to 0.')
