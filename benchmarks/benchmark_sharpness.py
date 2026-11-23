"""Synthetic sharp/blurred ordering and CPU timing; not threshold calibration.

Run: python -m benchmarks.benchmark_sharpness --repeats 5
"""
import argparse
import json
import time

import cv2
import numpy as np

from stash_ai.recognition.blur_quality import ALGORITHM, measure
from stash_ai.recognition.foreground_quality import laplacian


def run(repeats):
    y, x = np.indices((256, 256))
    pixels = (28+200*((x//16+y//16) % 2)).astype(np.float32)
    mask = (x-128)**2+(y-128)**2 < 105**2
    blur = cv2.GaussianBlur(pixels, (0, 0), 2)
    noise = np.random.default_rng(17).normal(0, 5, pixels.shape).astype(np.float32)
    cases = {'sharp_low_contrast': 128+.15*(pixels-128), 'sharp_high_contrast': pixels,
             'motion_horizontal': cv2.blur(pixels, (17, 1)),
             'defocus_gaussian': blur, 'defocus_plus_noise': blur+noise,
             'noise_only': 128+noise}
    result = []
    for name, image in cases.items():
        measured = measure(image, mask, diagnostics=False)
        started = time.perf_counter()
        for _ in range(repeats):
            measure(image, mask, diagnostics=False)
        elapsed = (time.perf_counter()-started)*1000/repeats
        result.append(dict(case=name, sharpness=measured['sharpness'],
                           legacy_laplacian=laplacian(image, mask)['sharpness'],
                           cpu_ms=elapsed))
    return dict(algorithm=ALGORITHM, scope='synthetic 256px foreground; uncalibrated', cases=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    print(json.dumps(run(args.repeats), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
