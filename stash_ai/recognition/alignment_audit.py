"""Explicit local, read-only alignment audit for VTT face thumbnails.

No facial embeddings, SQL state or performer changes. Source/aligned PNGs are
sensitive biometric data and are written only in a newly created private folder.
"""
from collections import Counter
from contextlib import closing
import json
import math
import os
from pathlib import Path
import statistics

from .aura import similarity_matrix
from .models import YuNetDetector, ensure_assets
from .quality import (
    ALIGNMENT_FACE_COLUMNS, ALIGNMENT_FACE_ROWS, MIN_ALIGNMENT_FACE_COVERAGE,
    MIN_ALIGNMENT_SOURCE_COVERAGE, assess, rejection,
)
from .visibility import LazyVisibility, VisibilityError, VisibilityGeometryError, cached as visibility_cached


def _coverage(width, height, matrix):
    """Same inverse-coordinate source coverage test as quality.alignable."""
    import cv2
    import numpy as np

    inverse = cv2.invertAffineTransform(matrix)
    rows, columns = np.ogrid[:112, :112]
    x = inverse[0, 0] * columns + inverse[0, 1] * rows + inverse[0, 2]
    y = inverse[1, 0] * columns + inverse[1, 1] * rows + inverse[1, 2]
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    face = valid[ALIGNMENT_FACE_ROWS, ALIGNMENT_FACE_COLUMNS]
    return valid, dict(
        source_coverage=float(valid.mean()),
        face_region_coverage=float(face.mean()),
        padding_fraction=float(1. - valid.mean()),
        face_region_padding_fraction=float(1. - face.mean()),
        source_coverage_gate=MIN_ALIGNMENT_SOURCE_COVERAGE,
        face_region_coverage_gate=MIN_ALIGNMENT_FACE_COVERAGE,
    )


def _diagnostics(image, detection):
    """Derive pixels used by ONNX aligners; does not invoke a recognizer."""
    import numpy as np

    x, y, width, height = (float(v) for v in detection.box)
    landmarks = np.asarray(detection.landmarks, dtype=np.float64)
    result = {
        'source_dimensions': [image.width, image.height],
        'face_box': [x, y, width, height],
        'face_min_dimension_pixels': min(width, height),
        'source_eye_distance_pixels': float(np.linalg.norm(landmarks[0] - landmarks[1])),
        'landmarks_source': landmarks.tolist(),
    }
    try:
        matrix = similarity_matrix(landmarks)
    except ValueError:
        result.update(alignment_valid=False, alignment_error='invalid_five_point_geometry')
        return result, None, None
    mask, stats = _coverage(image.width, image.height, matrix)
    scale = float(math.hypot(matrix[0, 0], matrix[1, 0]))
    result.update(stats)
    result.update(alignment_valid=(stats['source_coverage'] >= MIN_ALIGNMENT_SOURCE_COVERAGE
                                   and stats['face_region_coverage'] >= MIN_ALIGNMENT_FACE_COVERAGE),
                  affine_source_to_112=matrix.tolist(),
                  source_pixels_per_aligned_pixel=1. / scale,
                  aligned_eye_distance_pixels=float(
                      np.linalg.norm(
                          (matrix[:, :2] @ landmarks[0] + matrix[:, 2]) -
                          (matrix[:, :2] @ landmarks[1] + matrix[:, 2]))))
    return result, matrix, mask


def _png(image, path):
    # Path names are generated exclusively from validated numeric scene/cue IDs.
    # Exclusive creation avoids silently overwriting sensitive user assets.
    with path.open('xb') as output:
        image.save(output, 'PNG')
    if os.name == 'posix':
        path.chmod(0o600)


def _images(image, detection, matrix, mask, path):
    """Save one annotated source and an exact black-padded 112px model crop."""
    import cv2
    import numpy as np
    from PIL import Image
    from .person_audit import _draw

    source = image.convert('RGB')
    try:
        x, y, width, height = detection.box
        _draw(source, (x, y, x + width, y + height), detection.landmarks, face_only=True, labels=False)
        _png(source, path.with_name(path.name + '-source.png'))
    finally:
        source.close()

    if matrix is None:
        return ['-source.png']
    # The ONNX recognizers call warpAffine on BGR pixels and later reorder to
    # their respective input channels. RGB serialization is visual only.
    with image.convert('RGB') as converted:
        pixels = np.asarray(converted, dtype=np.uint8)
        bgr = np.ascontiguousarray(pixels[:, :, ::-1])
    aligned = cv2.warpAffine(bgr, matrix, (112, 112), borderValue=0.)
    output = Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    coverage = Image.fromarray((mask.astype('uint8') * 255), mode='L')
    try:
        _png(output, path.with_name(path.name + '-aligned.png'))
        _png(coverage, path.with_name(path.name + '-source-mask.png'))
    finally:
        output.close()
        coverage.close()
    return ['-source.png', '-aligned.png', '-source-mask.png']


def _sampled_indices(total, maximum):
    if total <= maximum:
        return set(range(total))
    if maximum == 1:
        return {total // 2}
    return {round(i * (total - 1) / (maximum - 1)) for i in range(maximum)}


def _private_directory(path):
    # Audit is explicitly opt-in and must never write into a pre-existing
    # directory where face images might accidentally be published or mixed.
    path = Path(path).expanduser()
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    if os.name == 'posix':
        path.chmod(0o700)
    return path


def _write_json(path, data):
    with path.open('x', encoding='utf-8') as output:
        json.dump(data, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write('\n')
    if os.name == 'posix':
        path.chmod(0o600)


def audit_scene(source, detector, config, scene_id, directory, *, maximum=None, cue_indices=None, visibility=None):
    """Export all passing VTT detections by default; optional cue/face caps are explicit."""
    from stash_ai.runtime.cleanup import closing

    total = len(source.cues)
    if cue_indices is not None:
        invalid = sorted(index for index in cue_indices if index >= total or index < 0)
        if invalid:
            raise ValueError(f'Scene {scene_id} has no selected cue index {invalid[0]}; valid range is 0..{total-1}.')
    selected = (set(cue_indices) if cue_indices is not None else
                None if maximum is None else _sampled_indices(total, maximum))
    policy = config.data['detection']
    from .pose_policy import measurement_required
    if measurement_required(config) and visibility is None:
        raise VisibilityError('Alignment audit requires FaceOcc for detection.min_visible_face_fraction.')
    stats, reasons, samples, sizes, coverages = Counter(), Counter(), [], [], []
    subdir = directory / ('scene-' + scene_id)
    subdir.mkdir(mode=0o700, exist_ok=False)
    with closing(source.frames(), name='close alignment audit frames') as frames:
        for index, cue, image in frames:
            with closing(image, name='close alignment audit image'):
                stats['cues_examined'] += 1
                faces = sorted(detector.detect(image), key=lambda item: tuple(item.row))
                for face_index, face in enumerate(faces):
                    stats['faces_detected'] += 1
                    quality = assess(image, face)
                    reason = ('face_detection_score' if face.score <= policy['score_threshold']
                              else rejection(quality, policy))
                    if reason is None and measurement_required(config):
                        try:
                            measurement = visibility.measure(image, face)
                        except VisibilityGeometryError:
                            reason = 'face_visibility_unmeasurable'
                        else:
                            if not visibility_cached(measurement):
                                raise VisibilityError('Alignment audit received invalid FaceOcc measurements.')
                            quality.update(measurement)
                            if quality['visible_face_fraction'] < policy['min_visible_face_fraction']:
                                reason = 'face_occluded'
                    reasons[reason or 'passed_detection_gates'] += 1
                    sizes.append(quality['pixels'])
                    if reason is not None:
                        stats['faces_rejected'] += 1
                        continue
                    stats['faces_passed_detection'] += 1
                    if ((selected is not None and index not in selected) or
                            (maximum is not None and len(samples) >= maximum)):
                        continue
                    measures, matrix, mask = _diagnostics(image, face)
                    if measures.get('source_coverage') is not None:
                        coverages.append(measures['source_coverage'])
                    name = f'cue-{index:05d}-face-{face_index:03d}'
                    suffixes = _images(image, face, matrix, mask, subdir / name)
                    person_input = None
                    if hasattr(detector, 'audit_input'):
                        from .person_collection import audit_input
                        person_input = audit_input(detector,image,face,subdir/name)
                    images = [f'scene-{scene_id}/{name}{suffix}' for suffix in suffixes]
                    if person_input:
                        images.extend(f'scene-{scene_id}/{person_input[key]}' for key in
                                      ('input_file', 'mask_file', 'source_file', 'annotated_input_file')
                                      if person_input.get(key))
                    samples.append({
                        'cue_index': index, 'seconds': float(cue.start),
                        'face_index': face_index, 'detection_score': float(face.score),
                        'quality': {key: value for key, value in quality.items()
                                    if key in ('pixels', 'pose_deviation', 'sharpness', 'exposure',
                                               'clipped_fraction', 'visible_face_fraction',
                                               'landmarks_inside', 'complete', 'alignment_valid')},
                        'detection_rejection': None,
                        'geometry': measures,
                        'person_input': person_input,
                        'images': images,
                    })
    return {
        'scene_id': scene_id, 'total_selected_vtt_cues': total,
        'sampling': ('explicit_indices' if cue_indices is not None else
                     'all_cues' if selected is None else 'evenly_spaced_indices'),
        'requested_sample_indices': None if selected is None else sorted(selected),
        'max_saved_faces': maximum,
        'counters': dict(stats), 'detection_rejections': dict(reasons),
        'sampled_faces': len(samples),
        'median_source_face_pixels_all_detections': statistics.median(sizes) if sizes else None,
        'median_alignment_source_coverage_saved': statistics.median(coverages) if coverages else None,
        'samples': samples,
    }


def execute(args, config, *, services=None, detector_factory=YuNetDetector):
    """Read-only: no FaceDatabase, no recognition run, no publication calls."""
    from .run import Services, instance_scope
    from .models import namespace
    from .scanning import scan_identity
    from .database import encoded
    from uuid import NAMESPACE_URL, uuid5
    from stash_ai.runtime.cleanup import closing

    services = services or Services()
    maximum = args.max_samples_per_scene
    if maximum is not None and maximum < 1:
        raise ValueError('--max-samples-per-scene must be a positive integer.')
    scenes = list(dict.fromkeys(args.scene_id or []))
    if not scenes or len(scenes) > 16:
        raise ValueError('Audit requires 1..16 explicit --scene-id arguments.')
    indices = None if not args.cue_index else sorted(set(args.cue_index))
    if indices is not None and any(i < 0 for i in indices):
        raise ValueError('--cue-index values must be nonnegative.')
    # Check selection before making even the output folder.
    directory = Path(args.output_dir).expanduser()
    if directory.exists():
        raise FileExistsError(f'Alignment audit output directory already exists: {directory}')
    if config.data['sapiensid']['enabled'] and detector_factory is YuNetDetector:
        from .sapiensid_backend import SapiensIDFaces
        detector_factory = lambda config, paths: SapiensIDFaces(config,paths,load_identity=False)
    url = os.getenv('STASH_URL', 'http://localhost:9999')
    key = os.getenv('STASH_API_KEY', '')
    with closing(services.client(url=url, api_key=key,
                                 timeout=config.data['execution']['timeout_seconds'])) as client:
        detector_key = config.data['models']['detector']
        from .pose_policy import measurement_required
        needs_visibility = measurement_required(config)
        keys = [detector_key] + (['faceocc'] if needs_visibility else [])
        paths = services.assets(config, keys=keys, write=lambda line: print(line))
        with closing(detector_factory(config, paths)) as detector, closing(
                LazyVisibility(config, paths, getattr(services, 'visibility', None)),
                name='close alignment audit visibility') as visibility:
            directory = _private_directory(directory)
            results = []
            model_namespace = namespace(config)
            scan_policy = scan_identity(config, model_namespace)
            scope = instance_scope(url)
            for scene_id in scenes:
                scene = client.scene(scene_id)
                with services.source(client, scene, config) as source:
                    result = audit_scene(source, detector, config, scene_id, directory,
                                         maximum=maximum, cue_indices=indices, visibility=visibility)
                    result['vtt_source_fingerprint'] = source.fingerprint
                    result['vtt_coverage'] = dict(source.coverage)
                    result['expected_recognition_scan_id'] = str(uuid5(
                        NAMESPACE_URL, encoded([scope, scene_id, model_namespace,
                                                 source.fingerprint, scan_policy])))
                    results.append(result)
            manifest = {
                'format': 2, 'kind': 'recognition_alignment_audit',
                'detector': config.data['models']['detector'],
                'recognizer_selected_but_not_loaded': config.data['models']['recognizer'],
                'embedding_namespace': model_namespace,
                'expected_scan_policy': scan_policy,
                'input_alignment': 'five-point ArcFace 112x112 shared by AuraFace, LVFace-B, AdaFace IR50',
                'detection_policy': dict(config.data['detection']),
                'limits': {'max_samples_per_scene': maximum, 'max_scenes': 16},
                'sample_filter': 'same detection score, quality and optional FaceOcc gates as recognition scan',
                'limitations': [
                    'The selected detector is executed again; no recognizer embeddings are generated.',
                    'FaceOcc is run when detection.min_visible_face_fraction is positive; no embedding or identity matching is run.',
                    'Images show the exact pre-normalization aligned BGR crop converted to RGB for inspection.',
                    'The inverse-mapped source mask follows quality.alignable; black pixels may also occur naturally.',
                    'The source landmarks, pose proxy and pixel sizes do not establish whether a face identity is correct.',
                    'Local PNGs are biometric material; keep the private folder out of source control and shared storage.',
                ],
                'scenes': results,
            }
            _write_json(directory / 'manifest.json', manifest)
            print(f'Alignment audit: {directory} | ' +
                  ' '.join(f"scene {r['scene_id']} saved={r['sampled_faces']}" for r in results))
            return 0
