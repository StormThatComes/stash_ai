"""Fresh, read-only full-cue face/pose detection exports for visual tuning."""
from dataclasses import replace
import os
from pathlib import Path

from stash_ai.runtime.cleanup import closing
from .alignment_audit import _png, _private_directory, _write_json
from .config import plain
from .detection_associations import REASON_LABELS, explain_associations, face_label
from .detection_callouts import draw_leaders, label_height, layout_faces
from .person_audit import BOX_COLOR, _draw
from .person_association import policy as association_policy


PERSON_COLORS = ((128, 0, 128), (0, 128, 0), (0, 0, 255))  # purple, green, blue
OVERFLOW_COLOR = (255, 0, 0)  # red for every subsequent person
UNMATCHED_FACE_BOX_COLOR = (255, 0, 0)


def _person_color(index):
    return PERSON_COLORS[index] if index < len(PERSON_COLORS) else OVERFLOW_COLOR


def _label(image, bounds, text, *, color):
    from PIL import ImageDraw
    painter = ImageDraw.Draw(image)
    box = painter.textbbox((0, 0), text)
    height = label_height(painter, text)
    x = max(0, min(bounds[0], image.width - (box[2] - box[0]) - 2))
    y = max(0, min(bounds[1] - height, image.height - height))
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
                person_colors = [_person_color(i) for i in range(len(people))]
                explanations = explain_associations(faces, people, detector.face_matches)
                face_labels = [face_label(i, face, match, explanation)
                               for i, (face, match, explanation) in enumerate(zip(
                                   faces, detector.face_matches, explanations))]
                layouts, output_size = layout_faces(image, faces, people, face_labels)
                name = f'scene-{scene_id}_cue-{index:06d}.png'
                record = dict(cue_index=index, start_seconds=float(cue.start), end_seconds=float(cue.end),
                              source_rect=list(cue.rect), source_dimensions=list(image.size),
                              output_dimensions=list(output_size), file=name, faces=[], people=[])
                from PIL import Image, ImageDraw
                with Image.new('RGB', output_size, (24, 24, 24)) as annotated:
                    annotated.paste(image, (0, 0))
                    draw_leaders(annotated, faces, layouts, image.size)
                    # Render pose over leaders, clipped to the source cue even
                    # when overflow face annotations need a larger canvas.
                    with annotated.crop((0, 0, *image.size)) as source_overlay:
                        for person_index, person in enumerate(people):
                            color = person_colors[person_index]
                            refinement = person.get('pose_refinement', {})
                            if refinement.get('box_expanded'):
                                ImageDraw.Draw(source_overlay).rectangle(refinement['detector_bounds'],
                                                                         outline=(160, 160, 160), width=1)
                            drawn = _draw(source_overlay, person['bounds'], person['keypoints'],
                                          box_color=color, point_color=color)
                            record['people'].append(dict(person_index=person_index, **person,
                                drawn_indices=drawn, annotation_color=list(color)))
                        for person_index, person in enumerate(people):
                            left, _, _, bottom = person['bounds']
                            _label(source_overlay, (left, bottom), f'P{person_index} {person["score"]:.3f}',
                                   color=person_colors[person_index])
                        annotated.paste(source_overlay, (0, 0))
                    for face_index, face in enumerate(faces):
                        layout = layouts[face_index]
                        person_index = detector.face_matches[face_index]
                        box_color = UNMATCHED_FACE_BOX_COLOR if person_index is None else BOX_COLOR
                        drawn = _draw(annotated, layout['bounds'], layout['landmarks'],
                                      face_only=True, box_color=box_color, point_color=BOX_COLOR)
                        record['faces'].append(dict(face_index=face_index, box=list(face.box), score=face.score,
                            landmarks=[list(point) for point in face.landmarks], drawn_indices=drawn,
                            annotation_color=list(BOX_COLOR), box_color=list(box_color),
                            display_offset=layout['offset'],
                            display_bounds=layout['bounds'], display_landmarks=layout['landmarks'],
                            person_index=person_index, association=explanations[face_index]))
                        _label(annotated, layout['bounds'], face_labels[face_index], color=BOX_COLOR)
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
                    format=3, kind='recognition_detection_audit', cue_index_base=0,
                    configuration={
                        'models': plain(config.data['models']),
                        'detection': plain(config.data['detection']),
                        'pose': plain(config.data['sapiensid']['pose']),
                        'sapiensid_enabled': config.data['sapiensid']['enabled'],
                        'collection_enabled': config.data['sapiensid']['collection']['enabled'],
                        'configured_max_cues': config.data['execution']['max_cues'],
                    },
                    execution=dict(max_cues=0, force_pose=True, embeddings=False),
                    annotations=dict(person_colors=[list(c) for c in PERSON_COLORS],
                        person_color_names=['purple', 'green', 'blue'], overflow_color=list(OVERFLOW_COLOR),
                        face_color=list(BOX_COLOR),
                        unmatched_face_box_color=list(UNMATCHED_FACE_BOX_COLOR),
                        association_reasons=REASON_LABELS,
                        face_person_association=association_policy(),
                        color_assignment='Per cue, pose colors follow person detection order; fourth and later '
                                         'people are red. Face points and labels are yellow. Face boxes are '
                                         'red when person_index is null, otherwise yellow; each face records box_color.',
                        pose_point_order='COCO17 (0..16)',
                        face_point_order='five face-detector landmarks (0..4)',
                        labels='F=index and score of face; -> P=index of matched person or unmatched; '
                               'unmatched faces add a reason below, with P/F candidate or competitor indices; '
                               'P=index and score of person; pose labels match boxes; face labels are yellow',
                        face_layout='Face boxes and landmarks are translated together to free space or a right '
                                    'gutter. Faint dashed leaders and hollow origins link valid source landmarks '
                                    'to display landmarks. Pose is drawn above leaders.',
                        coordinates='box/landmarks and person bounds/keypoints use cue-local source pixels; '
                                    'face boxes are xywh, bounds are xyxy. display_offset is [dx,dy]; '
                                    'display_bounds/display_landmarks use PNG pixels. Boxes are clipped to the '
                                    'source before translation; omitted display landmarks are [-1,-1]. '
                                    'The source stays unscaled at [0,0]; output_dimensions includes any gutter.'),
                    limitations=[
                        'Both detectors run on every cue, even when sapiensid/independent collection is disabled.',
                        'Configured score/NMS/result-count and pose-keypoint thresholds apply, including PMPose presence/visibility. Suppressed detections are unavailable.',
                        'All returned detections are drawn, without face/person quality, anchor or identity filtering.',
                        'With mask refinement, gray outlines show original expanded detector boxes; JSON records refinement decisions.',
                        'Missing/low-confidence/out-of-bounds keypoints are omitted, never inferred.',
                        'PNG pixels come from VTT sprites, not decoded video. No resizing or identity inference is performed.',
                        'Colors are local to each cue, not persistent identities across cues.',
                    ],
                    scene=result)
                _write_json(directory / 'manifest.json', manifest)
                report.console.finish_rendering()
                report.write(f'Detection audit: {directory} | scene {args.scene_id} | ' +
                             ' '.join(f'{key}={value}' for key, value in result['counts'].items()))
    return 0
