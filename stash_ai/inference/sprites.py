"""VTT cue sampling and bounded sprite decoding."""
from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
from typing import Iterator, TYPE_CHECKING
from urllib.parse import urljoin

from stash_ai.domain import Cue
from stash_ai.stash_api.transport import absolute_stash_url
from stash_ai.vtt import format_time, parse_vtt_cues

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def evenly_sample(items: list, count: int) -> list:
    if count <= 0 or len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def fetch_cues(app, stash, paths: dict) -> tuple[list[Cue], int]:
    if not paths.get("vtt"):
        raise ValueError("No VTT path; generate Stash sprites before running scene marker analysis.")
    content = stash.get_media(absolute_stash_url(paths["vtt"], base_url=app.STASH_URL))
    cues = parse_vtt_cues(content.decode("utf-8-sig"))
    # Resolve before sampling: a sample containing only one sheet from a
    # multi-sheet VTT must not be mistaken for a single-sheet scene.
    single_sheet = len({cue.image for cue in cues}) == 1
    cues = [
        Cue(
            cue.start,
            cue.end,
            absolute_stash_url(paths["sprite"], base_url=app.STASH_URL)
            if single_sheet and paths.get("sprite")
            else urljoin(absolute_stash_url(paths["vtt"], base_url=app.STASH_URL), cue.image),
            cue.rect,
        )
        for cue in cues
    ]
    return evenly_sample(cues, app.MAX_FRAMES_PER_SCENE), len(cues)


def iter_sprite_frames(stash, paths: dict, cues: list[Cue]) -> Iterator[tuple[Cue, PILImage]]:
    from PIL import Image

    sheets: OrderedDict[str, PILImage] = OrderedDict()
    try:
        for cue in cues:
            url = cue.image
            if url not in sheets:
                with Image.open(BytesIO(stash.get_media(url))) as opened:
                    sheets[url] = opened.convert("RGB")
                if len(sheets) > 2:
                    _, old = sheets.popitem(last=False)
                    old.close()
            sheets.move_to_end(url)
            sprite = sheets[url]
            x, y, width, height = cue.rect
            if x + width > sprite.width or y + height > sprite.height:
                raise ValueError(f"VTT rectangle exceeds sprite bounds at {format_time(cue.start)}.")
            frame = sprite.crop((x, y, x + width, y + height))
            try:
                yield cue, frame
            finally:
                frame.close()
    finally:
        for sprite in sheets.values():
            sprite.close()
