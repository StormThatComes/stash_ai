"""Frame-local explanations for PMPose duplicate selection, including mask-off audits."""
from contextlib import contextmanager
from contextvars import ContextVar


_CURRENT = ContextVar('pose_deduplication_audit', default=None)
MAX_PIXELS = 40_000_000


@contextmanager
def capture(image):
    reports = []
    token = _CURRENT.set((image, reports))
    try:
        yield reports
    finally:
        _CURRENT.reset(token)
        reports.clear()


def active(image):
    context = _CURRENT.get()
    return context is not None and context[0] is image


def publish(image, report):
    if active(image) and report is not None:
        reports = _CURRENT.get()[1]
        if reports:
            raise ValueError('Detection audit received multiple deduplication results for one cue.')
        reports.append(report)


def draw_removed(image, report):
    if report is None:
        return
    from PIL import ImageDraw
    painter = ImageDraw.Draw(image)
    for row in report['detections']:
        if row['suppressed_by'] is None:
            continue
        left, top, right, bottom = (int(v) for v in row['bounds'])
        left, top = max(0, left), max(0, top)
        right, bottom = min(image.width-1, right), min(image.height-1, bottom)
        for x in range(left, right+1, 10):
            for y in (top, bottom):
                painter.line((x, y, min(right, x+4), y), fill=(180, 180, 180))
        for y in range(top, bottom+1, 10):
            for x in (left, right):
                painter.line((x, y, x, min(bottom, y+4)), fill=(180, 180, 180))


@contextmanager
def append_explanations(image, report):
    if report is None:
        yield image
        return
    from PIL import Image, ImageDraw, ImageFont
    from .detection_sweep import _wrap
    detections = report['detections']
    lines = [f"Pose deduplication: {len(detections)} candidates -> {len(report['kept_indices'])} people. "
             'D=candidate after box NMS; P=output person. Gray dashed boxes=removed candidates.']
    for row in detections:
        target = row['person_index']
        if row['suppressed_by'] is not None:
            representative = row['suppressed_by']
            target = detections[representative]['person_index']
            status = f"removed -> D{representative}/P{target}"
        else:
            status = f'kept -> P{target}'
        text = f"D{row['detector_index']} {status} | core quality={row['quality_score']:.3f}"
        if 'detector_score' in row:
            text += f" | detector score={row['detector_score']:.3f}"
        lines.append(text)
    for pair in report['comparisons']:
        a, b = pair['detector_indices']
        text = (f"D{a}/D{b}: {pair['decision']} ({pair['reason']}) | IoU={pair['box_iou']:.3f} | "
                f"core={len(pair['matched_indices'])}/{len(pair['shared_indices'])} | "
                f"regions={','.join(pair['matched_regions']) or '-'}")
        if pair.get('max_normalized_distance') is not None:
            text += (f" | max point distance={pair['max_normalized_distance']:.3f} "
                     f"(match limit={report['settings']['max_point_distance']:.3f})")
        lines.append(text)
    if report['reason_counts'].get('no_box_overlap'):
        lines.append(f"{report['reason_counts']['no_box_overlap']} pairs kept: no box overlap.")
    if report['omitted_kept_pairs']:
        lines.append(f"{report['omitted_kept_pairs']} further kept pairs omitted; aggregate counts are in JSON.")
    font = ImageFont.load_default()
    width = max(640, image.width)
    lines = _wrap(lines, font, width-16)
    line_height = font.getbbox('Ag')[3] - font.getbbox('Ag')[1] + 6
    height = image.height + 16 + len(lines)*line_height
    if width*height > MAX_PIXELS:
        raise ValueError('Detection deduplication PNG exceeds 40,000,000 pixels; reduce pose.max_people_per_frame.')
    report['legend_offset'] = [0, image.height]
    with Image.new('RGB', (width, height), (24, 24, 24)) as canvas:
        canvas.paste(image, (0, 0))
        painter = ImageDraw.Draw(canvas)
        for index, text in enumerate(lines):
            painter.text((8, image.height+8+index*line_height), text, font=font, fill=(235, 235, 235))
        yield canvas
