"""WebVTT sprite cue parsing helpers."""
from __future__ import annotations

import re

from stash_ai.domain import Cue


TIMESTAMP_RE = re.compile(r"(?:(\d{2,}):)?(\d{2}):(\d{2})[.,](\d{3})")
RECT_RE = re.compile(r"(.*?)#xywh=(?:pixel:)?(\d+),(\d+),(\d+),(\d+)")


def timestamp_seconds(value: str) -> float:
    match = TIMESTAMP_RE.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid VTT timestamp: {value!r}")
    hours, minutes, seconds, milliseconds = match.groups()
    if int(minutes) >= 60 or int(seconds) >= 60:
        raise ValueError(f"Invalid VTT timestamp: {value!r}")
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

def format_time(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"

def parse_vtt_cues(vtt_text: str) -> list[Cue]:
    """Keep timing and image identity, including reused rectangles at new times."""
    text = vtt_text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    seen: set[Cue] = set()
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or re.match(r"^(NOTE|STYLE|REGION)(?:\s|$)", lines[0]):
            continue
        timing_lines = [i for i, line in enumerate(lines) if "-->" in line]
        if not timing_lines:
            continue  # WEBVTT header or metadata.
        if len(timing_lines) != 1:
            raise ValueError("Malformed VTT: expected one timing line per cue.")
        i = timing_lines[0]
        timing = re.fullmatch(r"(\S+)\s+-->\s+(\S+)(?:\s+.*)?", lines[i])
        if not timing:
            raise ValueError("Malformed VTT cue timing.")
        start, end = map(timestamp_seconds, timing.groups())
        if end <= start:
            raise ValueError("VTT cue end must be after its start.")
        payloads = [RECT_RE.fullmatch(line) for line in lines[i + 1:]]
        payloads = [match for match in payloads if match is not None]
        if len(payloads) != 1:
            raise ValueError("VTT cue is missing a single valid sprite rectangle.")
        image, *coordinates = payloads[0].groups()
        rect = tuple(map(int, coordinates))
        if rect[2] == 0 or rect[3] == 0:
            raise ValueError("VTT contains a zero-sized thumbnail.")
        cue = Cue(start, end, image, rect)
        if cue not in seen:
            cues.append(cue)
            seen.add(cue)
    cues.sort(key=lambda cue: (cue.start, cue.end, cue.image, cue.rect))
    if not cues:
        raise ValueError("No timestamped sprite thumbnails in VTT; generate Stash sprites first.")
    if any(b.start < a.end - 0.001 for a, b in zip(cues, cues[1:])):
        raise ValueError("Overlapping VTT thumbnail cues are ambiguous; regenerate the sprite.")
    return cues
