"""Public v2 YAML paths and their stable internal recognition settings.

Each row names a public group, its internal group, and leaves. An equals sign
renames a leaf (public=internal). Order also defines the printed document.
"""
from collections.abc import Mapping
import re

_GROUPS = (
    ('face.detector', 'models', 'backend=detector'),
    ('face.detector', 'detection', 'score_threshold nms_threshold max_faces_per_frame'),
    ('face.quality', 'detection',
     'min_face_pixels min_sharpness min_visible_face_fraction max_pose_deviation require_complete_face'),
    ('face.recognizer', 'models', 'backend=recognizer'),

    ('pose.detector', 'sapiensid.pose',
     'backend=detector score_threshold nms_threshold max_people_per_frame frame_batch_size=batch_size'),
    ('pose.detector.yolo', 'sapiensid.pose', 'path sha256'),
    ('pose.detector.dfine', 'sapiensid.pose', 'directory=dfine_directory'),
    ('pose.detector.rfdetr_xl', 'sapiensid.pose', 'directory=rfdetr_directory'),
    ('pose.estimator', 'sapiensid.pose', 'backend keypoint_score_threshold'),
    ('pose.estimator.vitpose_plus_large', 'sapiensid.pose', 'directory=vitpose_directory batch_size=vitpose_batch_size'),
    ('pose.estimator.sapiens', 'sapiensid.pose',
     'directory=sapiens_directory variant=sapiens_variant batch_size=sapiens_batch_size'),
    ('pose.estimator.pmpose', 'sapiensid.pose',
     'directory=pmpose_directory batch_size=pmpose_batch_size '
     'presence_threshold=pmpose_presence_threshold visibility_threshold=pmpose_visibility_threshold'),
    ('pose.refinement', 'sapiensid.pose.refinement',
     'enabled directory mode overlap_threshold min_keypoint_score min_presence min_visibility '
     'min_prompt_points max_prompt_points min_mask_score min_prompt_coverage max_foreign_coverage '
     'expand_boxes max_box_area_ratio max_side_expansion masked_identity_crop'),
    ('pose.deduplication', 'sapiensid.pose.deduplication',
     'enabled min_keypoint_score min_presence min_visibility min_shared_points max_point_distance '
     'min_match_fraction'),

    ('body', 'sapiensid', 'enabled'),
    ('body.encoder', 'sapiensid.model', 'variant source_directory checkpoint_directory model_yaml_sha256'),
    ('body.encoder', 'sapiensid.execution', 'batch_size'),
    ('body.collection', 'sapiensid.collection', 'enabled min_person_pixels min_sharpness min_keypoints max_overlap'),

    ('matching.face', 'recognition', 'metric match_threshold ambiguity_margin'),
    ('matching.face', 'clustering', 'cohesion_threshold deduplicate_detections'),
    ('matching.face.samples.minimum', 'clustering',
     'min_face_pixels=min_auxiliary_face_pixels min_sharpness=min_auxiliary_sharpness '
     'min_visible_face_fraction=min_auxiliary_visible_face_fraction'),
    ('matching.face.samples.references', 'clustering',
     'max_samples_per_person max_people_per_lookup max_reference_bytes'),
    ('matching.face.anchors', 'recognition', 'min_identity_support'),
    ('matching.face.anchors.quality', 'clustering',
     'min_face_pixels=min_anchor_face_pixels min_sharpness=min_anchor_sharpness '
     'min_visible_face_fraction=min_anchor_visible_face_fraction '
     'max_pose_deviation=max_anchor_pose_deviation'),
    ('matching.face.auxiliary', 'clustering', 'enabled=auxiliary_matching min_support=auxiliary_min_support'),
    ('matching.face.views', 'clustering',
     'prototypes_enabled=view_prototypes_enabled anchor_view_matching strict_appearance_matching '
     'anchor_outlier_quarantine context_view_enrichment corroborated_view_enrichment '
     'adaptive_view_witnesses match_threshold=view_match_threshold '
     'cohesion_threshold=view_cohesion_threshold ambiguity_margin=view_ambiguity_margin '
     'min_support=view_min_support attach_min_support=view_attach_min_support '
     'max_prototypes_per_person=max_view_prototypes_per_person max_witnesses=max_view_witnesses'),
    ('matching.face.same_scene', 'clustering', 'match_bonus=same_scene_match_bonus min_support=same_scene_min_support'),
    ('matching.face.cross_scene', 'recognition',
     'provisional=cross_scene_provisional min_merge_support cooccurrence_merge_threshold'),

    ('matching.body.identities', 'sapiensid.identities',
     'enabled tracklet_matching endpoint_matching match_threshold cohesion_threshold track_similarity_threshold scene_match_bonus scene_bonus_mode ambiguity_margin min_support attach_min_support '
     'min_appearance_distance'),
    ('matching.body.anchors', 'sapiensid.identities',
     'min_keypoints=anchor_min_keypoints min_person_pixels=anchor_min_person_pixels '
     'min_sharpness=anchor_min_sharpness'),
    ('matching.body.anchors.keypoint_rules', 'sapiensid.identities.anchor_keypoint_rules', 'all any'),
    ('matching.body.global', 'sapiensid.global_matching',
     'enabled match_threshold cohesion_threshold ambiguity_margin min_support max_references'),
    ('matching.hybrid', 'sapiensid.matching', 'face_floor person_threshold ambiguity_margin min_support'),
    ('matching.consolidation', 'sapiensid.consolidation', 'scope scene_ids max_scenes'),

    ('portraits', 'portraits', 'enabled'),
    ('portraits.selection', 'portraits', 'kind auto_precedence anchor_only max_candidates'),
    ('portraits.quality', 'portraits',
     'min_face_pixels min_sharpness min_visible_face_fraction max_pose_deviation min_exposure '
     'max_exposure max_clipped_fraction'),
    ('portraits.crop', 'portraits', 'aspect_ratio margin'),
    ('portraits.weights', 'portraits.weights',
     'visibility frontal resolution sharpness exposure neutral_expression open_eyes closed_mouth '
     'open_mouth'),
    ('portraits.source', 'portraits', 'kind=source ffmpeg time_offsets timeout_seconds max_video_bytes'),
    ('portraits.output', 'portraits', 'jpeg_quality'),

    ('execution.models', 'models', 'directory download_missing'),
    ('execution.gpu', 'models', 'device'),
    ('execution.gpu', 'execution', 'model_policy=gpu_model_policy log_vram cpu_residency_bytes sequential_scene_batch_size rfdetr_precision vitpose_precision vitpose_device_decode'),
    ('execution.scan', 'execution',
     'max_cues checkpoint_cues timeout_seconds cpu_workers=scan_cpu_workers queue_bytes=scan_queue_bytes '
     'detector_batch_size recognizer_batch_size prefetch_frames prefetch_scenes prefetch_scene_depth media_cache_bytes '
     'sprite_cache_bytes inference_cache_bytes'),
    ('execution.rebuild', 'execution',
     'disk_total_bytes disk_free_bytes sqlite_cache_bytes=rebuild_sqlite_cache_bytes scratch_bytes=rebuild_scratch_bytes scratch_directory=rebuild_scratch_directory '
     'parallel_scenes=rebuild_parallel_scenes storage=rebuild_storage workers=rebuild_workers native_threads=rebuild_native_threads '
     'cohort_backend=rebuild_cohort_backend nomination_device=rebuild_nomination_device '
     'prefetch_scenes=rebuild_prefetch_scenes profile=profile_rebuild memory_bytes=rebuild_memory_bytes '
     'cache_bytes=rebuild_cache_bytes matrix_cache_bytes=rebuild_matrix_cache_bytes '
     'queue_bytes=rebuild_queue_bytes checkpoint_mode=rebuild_checkpoint_mode checkpoint_scenes=rebuild_checkpoint_scenes '
     'checkpoint_bytes=rebuild_checkpoint_bytes replay_snapshot_bytes pixel_index_bytes derived_cache_bytes'),
    ('execution.rebuild.body', 'sapiensid.execution',
     'max_candidate_pairs max_witness_comparisons reference_bytes plan_bytes'),
    ('execution.solver', 'execution', 'compare_scene_strategies=solver_compare_scene_strategies'),
    ('execution.portraits', 'execution',
     'cpu_threads=portrait_cpu_threads faceocc_batch_size=portrait_faceocc_batch_size '
     'ranking_cache_bytes=portrait_ranking_cache_bytes prefetch=prefetch_portraits'),

    ('storage', 'storage', 'output_directory database max_crop_bytes crop_policy'),
    ('storage', 'sapiensid.storage', 'retain_person_crops'),
    ('storage.retention', 'storage.retention',
     'auto_prune keep_previous_scans keep_previous_person_contracts prune_obsolete_crops'),
    ('names', 'names', 'file'),
)


def _paths():
    result = {}
    for public, internal, leaves in _GROUPS:
        for field in leaves.split():
            key, _, renamed = field.partition('=')
            result[f'{public}.{key}'] = f'{internal}.{renamed or key}'
    return result


PUBLIC_PATHS = _paths()
INTERNAL_PATHS = {value: key for key, value in PUBLIC_PATHS.items()}
VITPOSE_BATCH = 'pose.estimator.vitpose_plus_large.batch_size'


def put(document, path, value):
    parts = path.split('.')
    for key in parts[:-1]:
        document = document.setdefault(key, {})
    document[parts[-1]] = value


def get(document, path):
    for key in path.split('.'):
        document = document[key]
    return document


def leaf_items(document, prefix=''):
    for key, value in document.items():
        if '.' in key:
            raise ValueError(f'Use nested YAML mappings for {prefix}{key}.')
        name = prefix + key
        if isinstance(value, Mapping) and value:
            yield from leaf_items(value, name + '.')
        else:
            yield name, value


POINT_FIELDS = ('min_keypoint_score', 'min_presence', 'min_visibility')
POINT_STAGES = ('refinement', 'deduplication')


def apply_v2(data, supplied):
    """Apply external overrides to a fresh internal defaults tree."""
    from .config import _number
    body = data['sapiensid']
    body['enabled'] = True
    body['pose'].update(detector='rfdetr_xl', backend='vitpose_plus_large')
    shared = dict.fromkeys(POINT_FIELDS, .5)
    shared_paths = {f'pose.point_quality.{field}': field for field in POINT_FIELDS}
    for key, value in leaf_items(supplied):
        if key == 'version':
            continue
        if key in shared_paths:
            _number(value, key, 0, 1)
            shared[shared_paths[key]] = value
            continue
        if key not in PUBLIC_PATHS:
            if value == {} and (key == 'pose.point_quality' or any(p.startswith(key + '.') for p in PUBLIC_PATHS)):
                continue
            raise ValueError(f'{key}: unknown v2 setting or expected a mapping.')
        put(data, PUBLIC_PATHS[key], value)
    source_body = supplied.get('body', {})
    if 'enabled' not in source_body.get('collection', {}):
        body['collection']['enabled'] = body['enabled']
    source_matching = supplied.get('matching', {}).get('body', {})
    if 'enabled' not in source_matching.get('identities', {}):
        body['identities']['enabled'] = body['collection']['enabled']
    if 'enabled' not in source_matching.get('global', {}):
        body['global_matching']['enabled'] = body['identities']['enabled']
    source_pose = supplied.get('pose', {})
    for stage in POINT_STAGES:
        for field in POINT_FIELDS:
            if source_pose.get(stage, {}).get(field) is None:
                body['pose'][stage][field] = shared[field]
    # V1 used one batch for frames and ViTPose crops. Canonicalize equal
    # values so equivalent configurations keep historical hashes and snapshots.
    pose = data['sapiensid']['pose']
    batch = pose.get('vitpose_batch_size', 4)
    _number(batch, VITPOSE_BATCH, 1, 16, integer=True)
    if batch == pose['batch_size']:
        pose.pop('vitpose_batch_size', None)
    else:
        pose['vitpose_batch_size'] = batch


def v2_document(data, *, source=b''):
    """Return a complete external document from effective internal settings."""
    from .config import parse_yaml, plain
    result = {'version': 2}
    for public, internal in PUBLIC_PATHS.items():
        if public == VITPOSE_BATCH:
            pose = data['sapiensid']['pose']
            value = pose.get('vitpose_batch_size', pose['batch_size'])
        else:
            value = get(data, internal)
        put(result, public, plain(value))
    # Preserve an explicit override even if it currently equals its inherited
    # value. A later edit to the shared threshold must still respect it.
    original = parse_yaml(source) if source else {}
    source_pose = original.get('pose', {}) if original.get('version') == 2 else {}
    pose = result['pose']
    shared = {}
    for field in POINT_FIELDS:
        first, second = (pose[stage][field] for stage in POINT_STAGES)
        default = source_pose.get('point_quality', {}).get(field, .5)
        if original.get('version') != 2 and first == second:
            default = first
        shared[field] = default
        for stage in POINT_STAGES:
            explicit = source_pose.get(stage, {}).get(field) is not None
            if not explicit and pose[stage][field] == default:
                pose[stage][field] = None
    result['pose'] = {'detector': pose['detector'], 'estimator': pose['estimator'],
                      'point_quality': shared, 'refinement': pose['refinement'],
                      'deduplication': pose['deduplication']}
    return result


def internal_path(path):
    """Resolve a v2 field, including an indexed regional-rule minimum."""
    if path in PUBLIC_PATHS:
        return PUBLIC_PATHS[path]
    rules = 'matching.body.anchors.keypoint_rules'
    if path == rules or path.startswith(rules + '.'):
        return 'sapiensid.identities.anchor_keypoint_rules' + path[len(rules):]
    return path


def public_path(path):
    rules = 'sapiensid.identities.anchor_keypoint_rules'
    if path == rules or path.startswith(rules + '.'):
        return 'matching.body.anchors.keypoint_rules' + path[len(rules):]
    return INTERNAL_PATHS[path]


def v2_patch(internal_patch):
    result = {}
    for path, value in leaf_items(internal_patch):
        put(result, INTERNAL_PATHS[path], value)
    return result


def v2_error(error):
    """Name invalid values using the paths the operator can edit in v2."""
    def replace_path(match):
        path = match.group()
        if path.startswith('sapiensid.identities.anchor_keypoint_rules'):
            return public_path(path)
        return INTERNAL_PATHS.get(path, path)
    return re.sub(r'[a-z_]+(?:\.[a-z_0-9]+)+', replace_path, str(error))
