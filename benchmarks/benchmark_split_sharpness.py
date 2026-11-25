"""Availability probe on existing photos; not a blur-labelled accuracy benchmark.

Run: python -m benchmarks.benchmark_split_sharpness
Head crops use a fixed 50px square and an ellipse, without alignment or FaceOcc.
Person boxes use an unmasked long-side resize, without SapiensID padding or SAM.
These exercise natural textures at representative sizes, not production crops.
"""
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from stash_ai.recognition.blur_quality import measure
from stash_ai.recognition.split_blur import METHOD

DATA = Path(__file__).parent/'data'/'dance_contact'


def samples(names=None):
    for name, case in json.loads((DATA/'labels.json').read_text()).items():
        if names is not None and name not in names:
            continue
        pixels = cv2.imread(str(DATA/(name+'.jpg')), cv2.IMREAD_GRAYSCALE)
        for index, person in enumerate(case['people']):
            for kind in ('face', 'body'):
                if kind == 'face':
                    if person.get('head') is None:
                        continue
                    x, y = person['head']
                    box = [x-25, y-25, x+25, y+25]
                else:
                    box = person['box']
                x0, y0, x1, y1 = map(int, box)
                crop = pixels[max(0, y0):y1, max(0, x0):x1]
                height, width = crop.shape
                scale = (112 if kind == 'face' else 384)/max(height, width)
                crop = cv2.resize(crop, (round(width*scale), round(height*scale)))
                mask = np.ones(crop.shape, bool)
                if kind == 'face':
                    yy, xx = np.indices(crop.shape)
                    mask = ((xx-56)/46)**2+((yy-56)/51)**2 < 1
                yield f'{name}-{index}-{kind}', kind, crop, mask


def run():
    rows = [dict(sample=name, kind=kind, quality=measure(pixels, mask))
            for name, kind, pixels, mask in samples()]
    return dict(method=METHOD, scope=__doc__, groups={kind: dict(
        total=sum(row['kind'] == kind for row in rows),
        reasons=dict(Counter(row['quality']['sharpness_split_reason'] for row in rows if row['kind'] == kind)))
        for kind in ('face', 'body')}, samples=rows)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, allow_nan=False))
