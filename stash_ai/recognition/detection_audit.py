"""Fresh, read-only full-cue face/pose detection exports for visual tuning."""
from dataclasses import replace
import os
from pathlib import Path

from stash_ai.runtime.cleanup import closing
from .alignment_audit import _png, _private_directory, _write_json
from .config import plain
from .person_audit import BOX_COLOR, FACE_COLORS, POSE_COLOR, _draw


def _label(image, bounds, text, *, color=BOX_COLOR):
    from PIL import ImageDraw
    painter = ImageDraw.Draw(image)
    box = painter.textbbox((0, 0), text)
    x = max(0, min(bounds[0], image.width - (box[2] - box[0]) - 2))
    y = max(0, min(bounds[1] - 14, image.height - 14))
    painter.text((x, y), text, fill=color, stroke_width=1, stroke_fill=(0, 0, 0))


def audit_scene(source, detector, scene_id, directory, *, progress=None):
    """Keep full source coordinates and all post-NMS detections, including faceless people."""
    frames = []
    counts = dict(cues_exported=0, faces_detected=0, people_detected=0, cues_without_detections=0)
    with closing(source.frames(), name='close detection audit frames') as images:
        for index, cue, image in images:
            # SpriteInput owns its images; the nested close also supports injected
            # sources and covers detection/export failures before the next yield.
            with closing(image, name='close detection audit image'), detector.prepared_frame(image):
                faces = detector.detect(image, force_pose=True)
                people = detector.frame_people
                name = f'scene-{scene_id}_cue-{index:06d}.png'
                record = dict(cue_index=index, start_seconds=float(cue.start), end_seconds=float(cue.end),
                              source_rect=list(cue.rect), source_dimensions=list(image.size),
                              file=name, faces=[], people=[])
                with image.convert('RGB') as annotated:
                    # People first: face landmarks remain visible in overlapping head regions.
                    for person_index, person in enumerate(people):
                        drawn = _draw(annotated, person['bounds'], person['keypoints'], box_color=POSE_COLOR)
                        record['people'].append(dict(person_index=person_index, **person, drawn_indices=drawn))
                    for face_index, face in enumerate(faces):
                        x, y, width, height = face.box
                        drawn = _draw(annotated, (x, y, x + width, y + height), face.landmarks, face_only=True)
                        record['faces'].append(dict(face_index=face_index, box=list(face.box), score=face.score,
                            landmarks=[list(point) for point in face.landmarks], drawn_indices=drawn,
                            person_index=detector.face_matches[face_index]))
                    for person_index, person in enumerate(people):
                        left, _, _, bottom = person['bounds']
                        _label(annotated, (left, bottom), f'P{person_index} {person["score"]:.3f}', color=POSE_COLOR)
                    for face_index, face in enumerate(faces):
                        _label(annotated, face.box, f'F{face_index} {face.score:.3f}')
                    _png(annotated, directory / name)
                frames.append(record)
                counts['cues_exported'] += 1
                counts['faces_detected'] += len(faces)
                counts['people_detected'] += len(people)
                counts['cues_without_detections'] += int(not faces and not people)
                if progress is not None:
                    progress.advance(faces=counts['faces_detected'], people=counts['people_detected'])
    if counts['cues_exported'] != len(source.cues):
        raise ValueError('Detection audit did not export every VTT cue.')
    return dict(scene_id=scene_id, counts=counts, frames=frames,
                vtt_source_fingerprint=source.fingerprint, vtt_coverage=dict(source.coverage))


def execute(args, config, *, services=None, detector_factory=None):
    """Load only detector models; never open identity state or publish to Stash."""
    from tqdm import tqdm
    from stash_ai.runtime.console import present, activity
    from stash_ai.runtime.report import RunReport
    from .progress import WorkProgress
    from .run import Services

    directory = Path(args.output_dir).expanduser()
    if directory.exists():
        raise FileExistsError(f'Detection audit output directory already exists: {directory}')
    services = services or Services()
    if detector_factory is None:
        from .sapiensid_backend import SapiensIDFaces
        detector_factory = lambda cfg, paths: SapiensIDFaces(cfg, paths, load_identity=False)
    # Always visit original VTT indices. Disable persistent HTTP caching so an
    # audit cannot modify the recognition database or its companion media DB.
    data = plain(config.data)
    data['execution'].update(max_cues=0, media_cache_bytes=0)
    source_config = replace(config, data=data)
    report = RunReport(args.progress_style == 'verbose', tqdm.write)
    url = os.getenv('STASH_URL', 'http://localhost:9999')
    with present(report, args.progress_style, status_position=1), closing(
            services.client(url=url, api_key=os.getenv('STASH_API_KEY', ''),
                            timeout=config.data['execution']['timeout_seconds']),
            name='close detection audit HTTP') as client:
        with activity(report, 'Load scene VTT and sprites'):
            scene = client.scene(args.scene_id)
            source = services.source(client, scene, source_config)
        with closing(source, name='close detection audit source'):
            if (not source.cues or not source.coverage['complete_sampling'] or
                    len(source.cues) != source.coverage['total_cues']):
                raise ValueError('Detection audit requires a nonempty, complete VTT cue source.')
            with activity(report, 'Load face and pose detectors'):
                paths = services.assets(config, keys=[config.data['models']['detector']], write=report.write)
                detector = detector_factory(config, paths)
            with closing(detector, name='close detection audit detectors'):
                directory = _private_directory(directory)
                with WorkProgress(report, 'Export detection audit', len(source.cues), unit='cue', key='cues') as progress:
                    result = audit_scene(source, detector, args.scene_id, directory, progress=progress)
                manifest = dict(
                    format=1, kind='recognition_detection_audit', cue_index_base=0,
                    configuration={
                        'models': plain(config.data['models']),
                        'detection': plain(config.data['detection']),
                        'pose': plain(config.data['sapiensid']['pose']),
                        'sapiensid_enabled': config.data['sapiensid']['enabled'],
                        'collection_enabled': config.data['sapiensid']['collection']['enabled'],
                        'configured_max_cues': config.data['execution']['max_cues'],
                    },
                    execution=dict(max_cues=0, force_pose=True, embeddings=False),
                    annotations=dict(box_color=list(BOX_COLOR), pose_color=list(POSE_COLOR),
                        face_box_color=list(BOX_COLOR), person_box_color=list(POSE_COLOR),
                        face_colors=[list(c) for c in FACE_COLORS], pose_point_order='COCO17 (0..16)',
                        face_point_order='five face-detector landmarks (0..4)',
                        labels='F=index and score of face; P=index and score of person; label colors match boxes',
                        coordinates='cue-local source pixels; face boxes are xywh, person bounds are xyxy'),
                    limitations=[
                        'Both detectors run on every cue, even when sapiensid/independent collection is disabled.',
                        'Configured score/NMS/result-count and pose-keypoint thresholds apply. Suppressed detections are unavailable.',
                        'All returned detections are drawn, without face/person quality, visibility, anchor or identity filtering.',
                        'Missing/low-confidence/out-of-bounds keypoints are omitted, never inferred.',
                        'PNG pixels come from VTT sprites, not decoded video. No resizing or identity inference is performed.',
                    ],
                    scene=result)
                _write_json(directory / 'manifest.json', manifest)
                report.console.finish_rendering()
                report.write(f'Detection audit: {directory} | scene {args.scene_id} | ' +
                             ' '.join(f'{key}={value}' for key, value in result['counts'].items()))
    return 0
