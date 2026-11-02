"""SAM overlays and optional per-person panels for detection audits."""
from contextlib import contextmanager
from dataclasses import dataclass
import math


MAX_PIXELS = 40_000_000


@dataclass(frozen=True)
class MaskOptions:
    view: str = 'overlay'
    opacity: float = .25
    show_rejected: bool = True

    def __post_init__(self):
        if self.view not in ('overlay', 'contour', 'panels', 'off'):
            raise ValueError('Mask view must be overlay, contour, panels or off.')
        if (isinstance(self.opacity, bool) or not isinstance(self.opacity, (int, float))
                or not math.isfinite(self.opacity) or not 0 <= self.opacity <= 1):
            raise ValueError('Mask opacity must be finite and in 0..1.')

    @classmethod
    def from_args(cls, args):
        return cls(getattr(args, 'mask_view', 'overlay'), getattr(args, 'mask_opacity', .25),
                   not getattr(args, 'hide_rejected_masks', False))

    def metadata(self):
        return dict(view=self.view, opacity=self.opacity, show_rejected=self.show_rejected,
                    colors='Same person palette as pose; colors do not identify people across cues/trials.',
                    contours='Solid for refined results; dashed for rejected results.',
                    prompts='White rings mark positive head/torso prompts sent to SAM.',
                    stages='raw: SAM output; cleaned: after speckle removal. Cleaned masks reach PMPose '
                           'only for refined or second_pass_core_drift results.',
                    coordinates='Mask pixels, prompts and panel source_dimensions use original cue coordinates. '
                                'Each detail panel has an offset within its frame PNG; in sweeps add the '
                                'frame source_offset to obtain combined PNG coordinates.')


def _stage(capture, index):
    stages = capture.people.get(index, {}) if capture is not None else {}
    return 'cleaned' if 'cleaned' in stages else 'raw' if 'raw' in stages else None


def _shown(record, options):
    return options.show_rejected or record.get('status') == 'refined'


def paint_mask(image, mask, color, *, opacity, rejected=False):
    """Blend in cue coordinates; include holes and every disconnected component."""
    import cv2
    import numpy as np
    from PIL import Image
    if mask.shape != (image.height, image.width):
        raise ValueError('Audit mask dimensions differ from the source cue.')
    pixels = mask.astype(np.uint8) * 255
    with Image.fromarray(pixels) as selection, Image.new('RGB', image.size, color) as tint:
        if opacity:
            with Image.blend(image, tint, opacity) as colored, Image.composite(colored, image, selection) as filled:
                image.paste(filled)
        if rejected:
            # Dash by distance along each boundary, not an image-space pattern
            # that can erase an entire diagonal edge. Include holes/components.
            edge = np.zeros_like(pixels)
            contours, _ = cv2.findContours(pixels, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                xy = contour.reshape(-1, 2)
                xy = xy[np.arange(len(xy)) % 10 < 5]
                edge[xy[:, 1], xy[:, 0]] = 255
        else:
            edge = pixels & ~cv2.erode(pixels, np.ones((3, 3), np.uint8),
                                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
        with Image.fromarray(edge) as outline:
            image.paste(tint, (0, 0), outline)


def draw_masks(image, people, capture, options, color_for):
    if options.view == 'off' or capture is None:
        return
    for index, person in enumerate(people):
        record = person.get('pose_refinement', {})
        stage = _stage(capture, index)
        if stage is not None and _shown(record, options):
            paint_mask(image, capture.mask(index, stage), color_for(index),
                       opacity=0 if options.view == 'contour' else options.opacity,
                       rejected=record.get('status') != 'refined')


def draw_prompts(image, people, options):
    from PIL import ImageDraw
    if options.view == 'off':
        return
    painter = ImageDraw.Draw(image)
    for person in people:
        record = person.get('pose_refinement', {})
        if not _shown(record, options):
            continue
        for x, y in record.get('prompt_points', []):
            painter.ellipse((x-6, y-6, x+6, y+6), outline=(0, 0, 0), width=3)
            painter.ellipse((x-5, y-5, x+5, y+5), outline=(255, 255, 255), width=2)


def _caption(index, record):
    status = record.get('status', 'unavailable')
    text = f'P{index} | SAM {status}'
    if 'mask_score' in record:
        text += f' | score={record["mask_score"]:.3f}'
    if record.get('reason'):
        text += ' | ' + record['reason']
    return text


@contextmanager
def append_details(image, annotated, people, capture, options, color_for):
    """Keep source and face geometry fixed; append legends/details below it."""
    from PIL import Image, ImageDraw, ImageFont
    from .detection_sweep import _wrap
    from .person_audit import _draw
    rows = [(i, person['pose_refinement']) for i, person in enumerate(people)
            if 'pose_refinement' in person] if options.view != 'off' else []
    if not rows:
        yield annotated, dict(view=options.view, people=[], panels=[])
        return
    font = ImageFont.load_default()
    width = max(annotated.width, 360, 2 * image.width if options.view == 'panels' else 0)
    line_height = font.getbbox('Ag')[3] - font.getbbox('Ag')[1] + 6
    descriptions = [((245, 245, 245), _wrap([
        'SAM masks: solid=accepted, dashed=rejected; white rings=prompt points.'
    ], font, width-16))]
    metadata = []
    for index, record in rows:
        stage = _stage(capture, index)
        shown = stage is not None and _shown(record, options)
        label = _caption(index, record)
        label += f' | {stage}' if shown else ' | mask hidden' if stage else ' | no mask'
        descriptions.append((color_for(index), _wrap([label], font, width-16)))
        metadata.append(dict(person_index=index, available_stages=list(capture.people.get(index, {}))
                             if capture is not None else [], displayed_stage=stage if shown else None,
                             second_pass_used=record.get('status') == 'refined' or
                             record.get('reason') == 'second_pass_core_drift'))
    height = annotated.height + 12 + sum(len(lines) * line_height for _, lines in descriptions)
    panels = []
    if options.view == 'panels':
        for index, record in rows:
            if not _shown(record, options):
                continue
            used = record.get('status') == 'refined' or record.get('reason') == 'second_pass_core_drift'
            labels = ['Raw SAM mask', 'Cleaned mask | ' + ('used by PMPose' if used else 'not used by PMPose')]
            headers = [_wrap([f'P{index} | ' + label], font, image.width-12) for label in labels]
            header_height = max(map(len, headers)) * line_height + 12
            for column, stage in enumerate(('raw', 'cleaned')):
                panels.append(dict(person_index=index, stage=stage,
                                   offset=[column * image.width, height + header_height],
                                   header_offset=[column * image.width, height], header_lines=headers[column],
                                   source_dimensions=list(image.size),
                                   available=capture is not None and stage in capture.people.get(index, {})))
            height += header_height + image.height + 12
    if width * height > MAX_PIXELS:
        raise ValueError('Detection mask PNG exceeds 40,000,000 pixels; use --mask-view overlay or off.')
    with Image.new('RGB', (width, height), (24, 24, 24)) as canvas:
        canvas.paste(annotated, (0, 0))
        painter = ImageDraw.Draw(canvas)
        y = annotated.height + 6
        for color, lines in descriptions:
            for line in lines:
                painter.text((8, y), line, font=font, fill=(245, 245, 245))
                painter.line((2, y, 2, y+line_height-2), fill=color, width=3)
                y += line_height
        for panel in panels:
            index = panel['person_index']
            record = people[index]['pose_refinement']
            x, y = panel['header_offset']
            for n, line in enumerate(panel['header_lines']):
                painter.text((x+6, y+6+n*line_height), line, font=font, fill=(245, 245, 245))
            if not panel['available']:
                x, y = panel['offset']
                painter.text((x+6, y+6), 'No mask for this stage', font=font, fill=(200, 200, 200))
                continue
            with image.convert('RGB') as detail:
                paint_mask(detail, capture.mask(index, panel['stage']), color_for(index),
                           opacity=options.opacity, rejected=record.get('status') != 'refined')
                draw_prompts(detail, [people[index]], options)
                ImageDraw.Draw(detail).rectangle(record['detector_bounds'], outline=(160, 160, 160), width=1)
                _draw(detail, people[index]['bounds'], people[index]['keypoints'],
                      box_color=color_for(index), point_color=color_for(index))
                canvas.paste(detail, tuple(panel['offset']))
        yield canvas, dict(view=options.view, people=metadata, panels=panels,
                           legend_offset=[0, annotated.height], source_dimensions=list(image.size))
