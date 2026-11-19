"""Restore missing inspection bitmaps from the prompts of measured SAM masks."""
from contextlib import closing
import json

import numpy as np

from .body_collection import quality
from .body_observations import enabled
from .config import digest
from .database import encoded
from .inference_cache import runtime_versions
from .mask_candidates import check_mask
from .pose_refinement import retain_quality_mask
from .progress import work
from .retained_body_masks import compress_mask


# Keep genuinely maskless observations out of the work queue. Missing prompts
# predate replay support. The scan, contract and cue restrict all lookups.
_GAPS = """ FROM body_observations b WHERE scan=? AND contract=?
    AND NOT EXISTS (SELECT 1 FROM body_quality_masks m WHERE m.observation=b.id)
    AND json_extract(quality,'$.quality.sharpness_method')='sam_mask'
    AND json_array_length(geometry,'$.pose_refinement.prompt_points')>0
    AND json_type(geometry,'$.pose_refinement.detector_bounds')='array'
    AND json_type(geometry,'$.pose_refinement.mask_pixels')='integer'
    AND json_extract(quality,'$.mask_replay_attempt') IS NOT ?"""


def _checked_result(image, row, config, points, raw, score):
    person = json.loads(row['geometry'])
    metadata = json.loads(row['quality'])
    recorded = metadata['quality']
    details = person['pose_refinement']
    settings = config.data['sapiensid']
    mask, box, _ = check_mask(raw, score, points, (), details['detector_bounds'],
                              settings['pose']['refinement'], image.size, replay=details.get('component_edits', ()),
                              replay_cuts=details.get('bridge_cuts', ()))
    bounds = person['bounds']
    if (mask is None or int(mask.sum()) != details['mask_pixels'] or
            (bounds != box and details.get('reason') != 'second_pass_core_drift')):
        return None

    retain_quality_mask(details, mask, bounds)
    packed = details.get('overlap_mask')
    if not packed:
        return None
    if 'foreground_hash' in recorded:
        details['encoder_mask'] = packed
    measured, _ = quality(image, person, (), settings['collection'])
    fields = ('sharpness_method', 'sharpness', 'sharpness_pixels', 'pixel_digest')
    if 'source_pixel_digest' in recorded:
        fields += ('source_pixel_digest',)
    if any(measured.get(key) != recorded.get(key) for key in fields):
        return None
    # Before bitmap retention, encoder-masked bodies had only a masked RGB
    # digest. Save the verified source crop digest so full-cue browsing works.
    if 'source_pixel_digest' in measured:
        recorded['source_pixel_digest'] = measured['source_pixel_digest']
    metadata['mask_replay'] = 'verified'
    metadata.pop('mask_replay_attempt', None)
    return compress_mask(packed), encoded(metadata)


def _restore_cue(collector, db, image, rows, attempt):
    prompts = []
    for row in rows:
        details = json.loads(row['geometry'])['pose_refinement']
        prompts.append((np.asarray(details['prompt_points'], dtype=np.float32),
                        np.asarray(details.get('negative_points', []), dtype=np.float32).reshape(-1, 2)))
    with collector.model.pose_refiner.mask_frame(image) as segmenter:
        results = []
        for row, (points, negative) in zip(rows, prompts):
            details = json.loads(row['geometry'])['pose_refinement']
            if 'mask_instance' in details:
                replay = getattr(segmenter, 'predict_instance', None)
                prediction = replay(details['mask_instance']) if callable(replay) else None
            elif details.get('retry_mask_choices') and callable(getattr(segmenter, 'predict_options', None)):
                options = segmenter.predict_options(points, negative)
                index = details.get('mask_variant', 0)
                prediction = options[index] if 0 <= index < len(options) else None
            else:
                prediction = segmenter.predict(points, negative)
            results.append(_checked_result(image, row, collector.config, points, *prediction)
                           if prediction is not None else None)
    # Recognition can use SAM's batched decoder for the first prompt set and
    # its scalar decoder for retries. Their edge pixels can differ on GPU.
    # Instance replay must never substitute another query through prompt selection.
    if callable(getattr(segmenter, 'predict_frames', None)):
        for index, result in enumerate(results):
            if result is None and 'mask_instance' not in json.loads(rows[index]['geometry'])['pose_refinement']:
                [predictions] = segmenter.predict_frames([(image, [prompts[index]])])
                [prediction] = predictions
                results[index] = _checked_result(image, rows[index], collector.config,
                                                 prompts[index][0], *prediction)
    with db:
        for row, result in zip(rows, results):
            if result is None:
                metadata = json.loads(row['quality'])
                metadata.update(mask_replay='measurement_mismatch', mask_replay_attempt=attempt)
                payload, metadata = None, encoded(metadata)
            else:
                payload, metadata = result
            # Inference runs without a write transaction. Only attach the result
            # if both the measured body and its missing bitmap still match.
            changed = db.execute(
                'UPDATE body_observations SET quality=? WHERE id=? AND quality=? AND geometry=? '
                'AND NOT EXISTS (SELECT 1 FROM body_quality_masks WHERE observation=?)',
                (metadata, row['id'], row['quality'], row['geometry'], row['id']))
            if changed.rowcount and payload is not None:
                db.execute('INSERT INTO body_quality_masks VALUES (?,?)', (row['id'], payload))


def restore_masks(collector, db, scan, source):
    config = collector.config
    settings = config.data['sapiensid']
    if not (enabled(config) and settings['storage']['retain_body_masks'] and settings['pose']['refinement']['enabled']):
        return
    attempt = digest(['sam-mask-replay-v2', config.data['models']['device'],
                      config.data['execution']['sam_precision'], runtime_versions()])
    params = (scan, collector.body_contract, attempt)
    cues = [r[0] for r in db.execute('SELECT DISTINCT cue' + _GAPS + ' ORDER BY cue', params)]
    if not cues:
        return
    expected = db.execute('SELECT input_sha FROM scans WHERE id=?', (scan,)).fetchone()[0]
    if source.fingerprint != expected:
        raise ValueError('VTT source changed before body mask replay.')
    wanted = set(cues)
    frames = source.frames_indices(cues) if hasattr(source, 'frames_indices') else source.frames(cues[0])
    collector._load()
    with closing(frames), work('Restore SAM body masks', len(cues), unit='cue') as progress:
        for cue, timing, image in frames:
            with image:
                if cue not in wanted:
                    continue
                rows = db.execute('SELECT id,geometry,quality,seconds' + _GAPS + ' AND cue=?',
                                  (*params, cue)).fetchall()
                if any(row['seconds'] != timing.start for row in rows):
                    raise ValueError('VTT cue changed before body mask replay.')
                if rows:
                    _restore_cue(collector, db, image, rows, attempt)
                progress.advance()
