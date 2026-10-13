"""Open lazy image headers under the explicit recognition pixel policy."""
from threading import RLock
import warnings


_open_lock = RLock()


def open_image(stream, limit):
    from PIL import Image
    # Recognition is a separate foreground invocation. Serialize its header
    # opens, restore Pillow's process default before decoding, and enforce our
    # exact limit before any conversion/allocation of the decoded image.
    with _open_lock, warnings.catch_warnings():
        previous = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = limit
            warnings.simplefilter('ignore', Image.DecompressionBombWarning)
            try:
                image = Image.open(stream)
            except Image.DecompressionBombError as error:
                raise ValueError('Recognition image exceeds its configured pixel limit.') from error
            if image.width * image.height > limit:
                image.close()
                raise ValueError('Recognition image exceeds its configured pixel limit.')
            return image
        finally:
            Image.MAX_IMAGE_PIXELS = previous
