"""Independent consumers of a validated person mask (no model imports)."""

FIELDS = {
    'sharpness': 'mask_sharpness',
    'body_encoder_input': 'masked_identity_crop',
    'pose_refinement': 'pose_second_pass',
    'keypoint_filtering': 'mask_keypoint_filtering',
    'box_expansion': 'expand_boxes',
    'overlap': 'mask_overlap',
    'body_admission': 'mask_body_admission',
    'contact_confirmation': 'mask_contact_confirmation',
}


def applications(settings, *, generic=False):
    second = settings.get('pose_second_pass')
    return {name: (settings.get('generic_second_pass', False) if generic else True)
            if name == 'pose_refinement' and second is None else
            settings.get(key, name not in ('body_encoder_input',))
            for name, key in FIELDS.items()}


def uses(person, name):
    """Old retained rows have the historical consumer policy."""
    policy = person.get('mask_usage', person.get('pose_refinement', {}).get('mask_usage', {}))
    if name == 'contact_confirmation' and name not in policy:
        return policy.get('overlap', True) and policy.get('body_admission', True)
    return policy.get(name, True)


def contact_disabled_settings(person):
    """Name the recorded controls, including historical coupled consumers."""
    policy = person.get('mask_usage', person.get('pose_refinement', {}).get('mask_usage', {}))
    names = ('contact_confirmation',) if 'contact_confirmation' in policy else ('overlap', 'body_admission')
    return ['pose.mask.apply_to.'+name for name in names if not policy.get(name, True)]


def admission_conflict(person):
    return uses(person, 'body_admission') and person.get('pose_refinement', {}).get('mask_conflict', False)
