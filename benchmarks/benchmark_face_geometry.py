"""Evaluate annotated box/profile ROIs and foreground sharpness calibration.

Input NPZ arrays: box_roi, profile_roi, visible_truth (N,H,W booleans),
sharpness, sharpness_normalized and blurred (N). All must describe held-out faces.
Run: python -m benchmarks.benchmark_face_geometry cases.npz --output report.json
"""
import argparse
import json
from pathlib import Path
import numpy as np


def evaluate(data):
    truth=np.asarray(data['visible_truth'],bool)
    result={}
    for name in ('box_roi','profile_roi'):
        roi=np.asarray(data[name],bool)
        if roi.shape!=truth.shape or truth.ndim!=3:raise ValueError('ROI/truth shapes must be N,H,W.')
        intersection=(roi&truth).sum(axis=(1,2))
        result[name]=dict(iou=(intersection/np.maximum(1,(roi|truth).sum(axis=(1,2)))).tolist(),
                          missed_visible_pixels=(truth&~roi).sum(axis=(1,2)).tolist(),
                          nonface_pixels=(roi&~truth).sum(axis=(1,2)).tolist())
    blurred=np.asarray(data['blurred'],bool)
    for name in ('sharpness','sharpness_normalized'):
        values=np.asarray(data[name],float)
        if values.shape!=blurred.shape or not np.isfinite(values).all():raise ValueError('Invalid sharpness labels.')
        result[name]=[dict(threshold=float(t),blur_recall=float(np.mean((values<t)[blurred])) if blurred.any() else None,
                          sharp_rejection=float(np.mean((values<t)[~blurred])) if (~blurred).any() else None)
                      for t in np.unique(values)]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cases');p.add_argument('--output',required=True)
    args=p.parse_args()
    with np.load(args.cases,allow_pickle=False) as data:report=evaluate(data)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
