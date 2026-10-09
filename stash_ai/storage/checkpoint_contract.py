"""Aggregation and frame-prefix checks shared by restore and offline inspection."""
from stash_ai.signatures import digest
import zlib


def retained_rows(identity, processed):
    if identity['aggregation'] == 'max':
        return min(1, processed)
    top_k = identity['top_k']
    if identity['aggregation'] != 'topk_mean' or type(top_k) is not int or top_k < 1:
        raise ValueError('Invalid checkpoint aggregation provenance.')
    return min(top_k, processed)


def registered_contract(db, scope, scene, model, checkpoint, *, table='inference_identities'):
    from stash_ai.inference.provenance import decode_identity
    from stash_ai.evidence.temporal.statistics import CHECKPOINT_FORMAT
    for signature, data in db.execute(
            f'SELECT signature,data FROM {table} WHERE scope=? AND scene=? AND model=?',
            (scope, str(scene), model)):
        try:
            identity, _ = decode_identity(data, signature)
        except (KeyError, TypeError, ValueError, UnicodeError, zlib.error):
            continue
        for kinds in (['scene'], ['markers'], ['scene', 'markers']):
            for analysis in (None, CHECKPOINT_FORMAT):
                if digest(dict(input=signature, kinds=kinds, scene_frames=1,
                               scene_analysis=analysis)) == checkpoint:
                    return identity
    return None


def validate_chunks(chunks, processed):
    from stash_ai.db.payloads import valid_frames
    frames, previous = [], 0
    for end, chunk in chunks:
        if (type(end) is not int or not valid_frames(chunk, checkpoint=True)
                or end != previous + len(chunk) or end > processed):
            raise ValueError('Invalid checkpoint frame chunk progress.')
        frames.extend(chunk)
        previous = end
    if previous != processed or not valid_frames(frames, checkpoint=True):
        raise ValueError('Incomplete checkpoint frame prefix; use --restart.')
    return frames
