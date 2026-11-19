"""Render recorded stages lazily, in source cue coordinates, without model calls."""


def render(source, phase):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from .detection_audit import _person_color
    from .detection_masks import paint_mask
    from .detection_sweep import _wrap
    from .person_audit import _draw

    font = ImageFont.load_default()
    width = max(source.width, 640)
    lines = [phase.title, 'White rings: positive prompts. Red crosses: negative prompts. '
             'Dashed masks: rejected. Raw points precede output quality filters.']
    lines += [f'{person.label}: {person.note}' for person in phase.people if person.note]
    lines = _wrap(lines, font, width - 16)
    line_height = font.getbbox('Ag')[3] - font.getbbox('Ag')[1] + 6
    output = Image.new('RGB', (width, source.height + 12 + len(lines) * line_height), (24, 24, 24))
    with source.convert('RGB') as overlay:
        for i, person in enumerate(phase.people):
            if person.mask is not None:
                shape, packed = person.mask
                pixels = np.unpackbits(np.frombuffer(packed, dtype=np.uint8),
                                       count=shape[0]*shape[1]).reshape(shape).astype(bool)
                paint_mask(overlay, pixels, _person_color(i), opacity=.25, rejected=person.rejected)
        painter = ImageDraw.Draw(overlay)
        for i, person in enumerate(phase.people):
            color = _person_color(i)
            _draw(overlay, person.bounds, person.points, box_color=color, point_color=color)
            for x, y in person.positive:
                painter.ellipse((x-5, y-5, x+5, y+5), outline='white', width=2)
            for x, y in person.negative:
                painter.line((x-5, y-5, x+5, y+5), fill=(255, 70, 70), width=2)
                painter.line((x-5, y+5, x+5, y-5), fill=(255, 70, 70), width=2)
            x, y = person.bounds[:2]
            painter.text((max(0, x+2), max(0, y+2)), person.label, fill=color,
                         stroke_width=1, stroke_fill='black')
        output.paste(overlay)
    painter = ImageDraw.Draw(output)
    for i, line in enumerate(lines):
        painter.text((8, source.height+6+i*line_height), line, font=font, fill=(240, 240, 240))
    return output
