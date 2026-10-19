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
    from PIL import Image, ImageDraw

    source = image.convert('RGB')
    try:
        painter = ImageDraw.Draw(source)
        x, y, width, height = detection.box
        painter.rectangle((x, y, x + width, y + height), outline=(255, 220, 0), width=2)
        for number, (px, py) in enumerate(detection.landmarks):
            color = [(0, 240, 255), (0, 240, 255), (255, 80, 80),
                     (150, 255, 80), (150, 255, 80)][number]
            painter.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
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


def audit_scene(source, detector, config, scene_id, directory, *, maximum=32, cue_indices=None):
    """Scan all VTT cues but serialize only evenly spaced (or explicit) cues."""
    from stash_ai.runtime.cleanup import closing

    total = len(source.cues)
    if cue_indices is not None:
        invalid = sorted(index for index in cue_indices if index >= total or index < 0)
        if invalid:
            raise ValueError(f'Scene {scene_id} has no selected cue index {invalid[0]}; valid range is 0..{total-1}.')
    selected = set(cue_indices) if cue_indices is not None else _sampled_indices(total, maximum)
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
                    reason = ('face_detection_score' if face.score <= config.data['detection']['score_threshold']
                              else rejection(quality, config.data['detection']))
                    reasons[reason or 'passed_nonvisibility_gates'] += 1
                    sizes.append(quality['pixels'])
                    # Preserve the actual detector result, including rejected
                    # faces, so alignment can be investigated independently
                    # from the inference and visibility gates.
                    if index not in selected or len(samples) >= maximum:
                        continue
                    measures, matrix, mask = _diagnostics(image, face)
                    if measures.get('source_coverage') is not None:
                        coverages.append(measures['source_coverage'])
                    name = f'cue-{index:05d}-face-{face_index:03d}'
                    suffixes = _images(image, face, matrix, mask, subdir / name)
                    samples.append({
                        'cue_index': index, 'seconds': float(cue.start),
                        'face_index': face_index, 'detection_score': float(face.score),
                        'quality': {key: value for key, value in quality.items()
                                    if key in ('pixels','pose_deviation','sharpness',
                                               'landmarks_inside','complete','alignment_valid')},
                        'nonvisibility_rejection': reason,
                        'geometry': measures,
                        'images': [f'scene-{scene_id}/{name}{suffix}' for suffix in suffixes],
                    })
    return {
        'scene_id': scene_id, 'total_selected_vtt_cues': total,
        'sampling': 'explicit_indices' if cue_indices is not None else 'evenly_spaced_indices',
        'requested_sample_indices': sorted(selected), 'max_saved_faces': maximum,
        'counters': dict(stats), 'nonvisibility_rejections': dict(reasons),
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
    if not 1 <= maximum <= 128:
        raise ValueError('--max-samples-per-scene must be between 1 and 128.')
    scenes = list(dict.fromkeys(args.scene_id or []))
    if not scenes or len(scenes) > 16:
        raise ValueError('Audit requires 1..16 explicit --scene-id arguments.')
    indices = None if not args.cue_index else sorted(set(args.cue_index))
    if indices is not None and (len(indices) > 128 or any(i < 0 for i in indices)):
        raise ValueError('At most 128 nonnegative --cue-index values are allowed.')
    # Check selection before making even the output folder.
    directory = Path(args.output_dir).expanduser()
    if directory.exists():
        raise FileExistsError(f'Alignment audit output directory already exists: {directory}')
    url = os.getenv('STASH_URL', 'http://localhost:9999')
    key = os.getenv('STASH_API_KEY', '')
    with closing(services.client(url=url, api_key=key,
                                 timeout=config.data['execution']['timeout_seconds'])) as client:
        paths = services.assets(config, keys=['yunet'], write=lambda line: print(line))
        with closing(detector_factory(config, paths)) as detector:
            directory = _private_directory(directory)
            results = []
            model_namespace = namespace(config)
            scan_policy = scan_identity(config, model_namespace)
            scope = instance_scope(url)
            for scene_id in scenes:
                scene = client.scene(scene_id)
                with services.source(client, scene, config) as source:
                    result = audit_scene(source, detector, config, scene_id, directory,
                                         maximum=maximum, cue_indices=indices)
                    result['vtt_source_fingerprint'] = source.fingerprint
                    result['vtt_coverage'] = dict(source.coverage)
                    result['expected_recognition_scan_id'] = str(uuid5(
                        NAMESPACE_URL, encoded([scope, scene_id, model_namespace,
                                                 source.fingerprint, scan_policy])))
                    results.append(result)
            manifest = {
                'format': 1, 'kind': 'recognition_alignment_audit',
                'detector': 'yunet',
                'recognizer_selected_but_not_loaded': config.data['models']['recognizer'],
                'embedding_namespace': model_namespace,
                'expected_scan_policy': scan_policy,
                'input_alignment': 'five-point ArcFace 112x112 shared by AuraFace, LVFace-B, AdaFace IR50',
                'detection_policy': dict(config.data['detection']),
                'limits': {'max_samples_per_scene': maximum, 'max_scenes': 16},
                'limitations': [
                    'YuNet face detection is executed again; no recognizer embeddings are generated.',
                    'The visibility model is not run: passed_nonvisibility_gates is not an accepted recognition sample.',
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
