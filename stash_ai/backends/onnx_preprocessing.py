"""ONNX image preprocessing for WD, DTQ and Oppai; no model/session IO."""
from __future__ import annotations

from typing import Any


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)



def prepare_oppai(image: Any, size: int = 448) -> tuple[Any, Any]:
    """Author's V1.1 letterbox: no upscaling; True in mask means padding."""
    import numpy as np
    from PIL import Image, ImageOps
    from contextlib import ExitStack

    with ExitStack() as owned:
        transposed = ImageOps.exif_transpose(image)
        owned.callback(transposed.close)
        if transposed.mode in ("RGBA", "LA") or "transparency" in transposed.info:
            rgb = Image.new("RGB", transposed.size, (114, 114, 114))
            owned.callback(rgb.close)
            rgba = transposed.convert("RGBA")
            owned.callback(rgba.close)
            alpha = rgba.getchannel("A")
            owned.callback(alpha.close)
            foreground = rgba.convert("RGB")
            owned.callback(foreground.close)
            rgb.paste(foreground, mask=alpha)
        else:
            rgb = transposed.convert("RGB")
            owned.callback(rgb.close)

        scale = min(size / rgb.width, size / rgb.height, 1.0)
        width, height = max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))
        resized = (rgb.resize((width, height), Image.Resampling.BILINEAR)
                   if (width, height) != rgb.size else rgb)
        if resized is not rgb:
            owned.callback(resized.close)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        top, left = (size - height) // 2, (size - width) // 2
        canvas[top:top + height, left:left + width] = np.asarray(resized, dtype=np.uint8)
        mask = np.ones((size, size), dtype=np.bool_)
        mask[top:top + height, left:left + width] = False
        pixels = ((canvas.astype(np.float32) / np.float32(255.0))
                  - np.float32(0.5)) / np.float32(0.5)
        return np.ascontiguousarray(pixels.transpose(2, 0, 1)), mask


def prepare_wd(image: Any, size: int) -> Any:
    import numpy as np
    from PIL import Image
    from contextlib import ExitStack

    with ExitStack() as owned:
        if image.mode == 'RGB' and 'transparency' not in image.info:
            rgb = image  # borrowed, never close
        else:
            rgba = image.convert("RGBA")
            owned.callback(rgba.close)
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            owned.callback(background.close)
            background.alpha_composite(rgba)
            rgb = background.convert("RGB")
            owned.callback(rgb.close)
        side = max(rgb.size)
        padded = Image.new("RGB", (side, side), (255, 255, 255))
        owned.callback(padded.close)
        padded.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
        resized = padded.resize((size, size), Image.Resampling.BICUBIC) if side != size else padded
        if resized is not padded:
            owned.callback(resized.close)
        return np.ascontiguousarray(np.asarray(resized, dtype=np.float32)[:, :, ::-1])


def prepare_dtq(image: Any, size: int) -> Any:
    import numpy as np
    from PIL import Image
    from contextlib import ExitStack

    with ExitStack() as owned:
        rgb = image.convert("RGB")
        owned.callback(rgb.close)
        scale = size / max(rgb.size)
        width, height = max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))
        resized = rgb.resize((width, height), Image.Resampling.BILINEAR)
        owned.callback(resized.close)
        padded = Image.new("RGB", (size, size), (0, 0, 0))
        owned.callback(padded.close)
        padded.paste(resized, ((size - width) // 2, (size - height) // 2))
        data = np.asarray(padded, dtype=np.float32).transpose(2, 0, 1) / np.float32(255.0)
        data = ((data - np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None])
                / np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None])
        return np.ascontiguousarray(data, dtype=np.float32)
