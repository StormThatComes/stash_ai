"""Shared Pillow input policy for native-size Stash images."""


def open_image(stream):
    from PIL import Image
    # Stash sheets use their native dimensions in both tagging and recognition.
    # This application-wide policy also covers Pillow's later crop checks, so
    # opening a sheet and using its pixels share the same size policy. Disable
    # only the decompression-bomb pixel threshold; unrelated warnings stay on.
    Image.MAX_IMAGE_PIXELS = None
    return Image.open(stream)
