"""Lossless storage of crop-local SAM quality masks, independent of crop retention."""
import base64
import zlib

SCHEMA = '''
CREATE TABLE body_quality_masks (
 observation TEXT NOT NULL PRIMARY KEY REFERENCES body_observations(id) ON DELETE CASCADE,
 mask BLOB NOT NULL
);
'''


def compress_mask(packed):
    # Recognition already validated this packed, row-major bitstream.
    return zlib.compress(base64.b64decode(packed, validate=True), level=1)


def mask_image(payload, width, height):
    """Decode exactly the expected crop, including a partial final byte."""
    import numpy as np
    from PIL import Image

    count = width * height
    if width <= 0 or height <= 0 or (Image.MAX_IMAGE_PIXELS is not None and count > Image.MAX_IMAGE_PIXELS):
        raise ValueError('Invalid stored mask dimensions.')
    size = (count + 7) // 8
    decoder = zlib.decompressobj()
    packed = decoder.decompress(payload, size + 1)
    if len(packed) != size or not decoder.eof or decoder.unused_data:
        raise ValueError('Stored mask differs from the crop dimensions.')
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=count)
    return Image.fromarray(bits.reshape(height, width) * 70)
