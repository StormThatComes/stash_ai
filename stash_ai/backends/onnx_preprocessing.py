"""ONNX image preprocessing for WD, DTQ and Oppai; no model/session IO."""
from __future__ import annotations

from typing import Any


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def prepare_oppai(image: Any, size: int = 448) -> tuple[Any, Any]:
    """Author's V1.1 letterbox: no upscaling; True in mask means padding."""
    import numpy as np
    from PIL import Image, ImageOps

    transposed = ImageOps.exif_transpose(image)
    if transposed.mode in ("RGBA", "LA") or "transparency" in transposed.info:
        rgb = Image.new("RGB", transposed.size, (114, 114, 114))
        rgba = transposed.convert("RGBA")
        alpha = rgba.getchannel("A")
        foreground = rgba.convert("RGB")
        rgb.paste(foreground, mask=alpha)
        foreground.close()
        alpha.close()
        rgba.close()
    else:
        rgb = transposed.convert("RGB")

    scale = min(size / rgb.width, size / rgb.height, 1.0)
    width, height = max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))
    resized = (
        rgb.resize((width, height), Image.Resampling.BILINEAR)
        if (width, height) != rgb.size else rgb
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - height) // 2, (size - width) // 2
    canvas[top:top + height, left:left + width] = np.asarray(resized, dtype=np.uint8)
    mask = np.ones((size, size), dtype=np.bool_)
    mask[top:top + height, left:left + width] = False
    pixels = (
        canvas.astype(np.float32) / np.float32(255.0) - np.float32(0.5)
    ) / np.float32(0.5)

    if resized is not rgb:
        resized.close()
    rgb.close()
    transposed.close()
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)), mask


def prepare_wd(image: Any, size: int) -> Any:
    import numpy as np
    from PIL import Image

    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    rgb = background.convert("RGB")
    side = max(rgb.size)
    padded = Image.new("RGB", (side, side), (255, 255, 255))
    padded.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
    resized = padded.resize((size, size), Image.Resampling.BICUBIC) if side != size else padded
    result = np.ascontiguousarray(np.asarray(resized, dtype=np.float32)[:, :, ::-1])
    for opened in (rgba, background, rgb, padded, resized):
        opened.close()
    return result


def prepare_dtq(image: Any, size: int) -> Any:
    import numpy as np
    from PIL import Image

    rgb = image.convert("RGB")
    scale = size / max(rgb.size)
    width, height = max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))
    resized = rgb.resize((width, height), Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (size, size), (0, 0, 0))
    padded.paste(resized, ((size - width) // 2, (size - height) // 2))
    data = np.asarray(padded, dtype=np.float32).transpose(2, 0, 1) / np.float32(255.0)
    data = (
        data - np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    ) / np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
    for opened in (rgb, resized, padded):
        opened.close()
    return np.ascontiguousarray(data, dtype=np.float32)
