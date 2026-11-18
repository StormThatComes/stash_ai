"""Bounded raw neighbour evidence, independent of inference/checkpoint windows."""
from contextlib import closing
from .temporal_prompts import capture
from .prompt_recovery import eligible


def contexts(owner, frames, predictions):
    result = {}
    for (image, boxes), raw in zip(frames, predictions):
        valid, _ = eligible(*raw, boxes, owner.settings)
        value = capture(image, boxes, raw[0], valid)
        if value:
            result[value[0][:2]] = value
    for image, _ in frames:
        signature = getattr(image, '_recognition_cue_signature', None)
        source_model = getattr(image, '_recognition_pose_source', None)
        if not signature or not source_model:
            continue
        source, model = source_model
        scan = image._recognition_scan
        wanted = [i for i in (signature[0]-1, signature[0]+1)
                  if 0 <= i < len(source.cues) and (scan, i) not in result]
        if not wanted:
            continue
        stream = (source.frames_indices(wanted) if hasattr(source, 'frames_indices')
                  else source.frames(min(wanted)))
        with closing(stream):
            for cue, _, neighbour in stream:
                with neighbour:
                    if cue not in wanted:
                        continue
                    from .perspective_shots import advance
                    if cue > signature[0]:
                        current = advance(image, signature[0], None, evidence=True)
                        state = advance(neighbour, cue, (signature[0],signature[1],*current[2:]), evidence=True)
                    else:
                        before = advance(neighbour, cue, None, evidence=True)
                        after = advance(image, signature[0], (cue,signature[1],*before[2:]), evidence=True)
                        state = (cue,signature[1] if after[4] else None,*before[2:])
                    neighbour._recognition_scan = scan
                    neighbour._recognition_cue_signature = state
                    if state[1] is None or state[1] != signature[1]:
                        result[scan,cue] = None
                        continue
                    deferred = []
                    model.detect(neighbour, faces=[], force_pose=True, _defer=deferred)
                    boxes = deferred[0][2]
                    raw = owner.pose.predict_many([(neighbour,boxes)])[0]
                    settings = model.config.data['sapiensid']['pose']['deduplication']
                    if settings['enabled']:
                        from .pose_deduplication import select
                        keep = select(boxes, raw, settings)['kept_indices']
                        boxes = [boxes[i] for i in keep]
                        raw = raw[0][keep], raw[1][keep], {k:v[keep] for k,v in raw[2].items()}
                    valid, _ = eligible(*raw, boxes, owner.settings)
                    result[scan,cue] = capture(neighbour,boxes,raw[0],valid)
                if cue >= max(wanted):
                    break
    return result
