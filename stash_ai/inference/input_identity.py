"""Stable cue manifests and input digests shared by every inference path."""
from stash_ai.signatures import digest


def cue_manifest(cues):
    return [[cue.start, cue.end, cue.image, list(cue.rect)] for cue in cues]


def frame_signature(manifest, hashes):
    return digest({'cues': manifest, 'media': hashes})


def inference_signature(model_identity, manifest, hashes):
    return digest({'model': model_identity, 'cues': manifest, 'media': hashes})
