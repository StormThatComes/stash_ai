"""Pinned ViTPose++ Large COCO17 assets; no inference dependencies."""

REPOSITORY = 'usyd-community/vitpose-plus-large'
REVISION = 'e211df377e49c89dba508ab3e83ddef13c0832b4'
BASE_URL = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/'
INFERENCE = 'vitpose-plus-large-coco17-topdown-fp32-v1'
# Sizes and SHA256s from the pinned Hub files (LFS SHA256 for the weights).
ASSETS = {
    'config.json': (2322, '4e07052e9e501fbabc2b0a9dd4a70bab094e3e5426b3b5ce62776443f3af208d'),
    'preprocessor_config.json': (363, '9b11cadc98c30b968a70cc1658ce1fbd74b721b0f21402a2ff8bc1dc9d1474a0'),
    'model.safetensors': (1737705492, '8b249f74fca7c90b94adb1ebcdbb32cb424fb4fb3c8c40f54a050ccce972643f'),
}


def contract():
    return dict(repository=REPOSITORY, revision=REVISION, inference=INFERENCE,
                dataset_index=0, assets={name: checksum for name, (_, checksum) in ASSETS.items()})
