"""Source-pixel boundary sensitivity diagnostic, not a calibrated probability."""
import numpy as np


def boundary_fraction(mask):
    if not mask.any():
        return 1.
    interior=np.zeros_like(mask)
    interior[1:-1,1:-1]=(mask[1:-1,1:-1]&mask[:-2,1:-1]&mask[2:,1:-1]
                        &mask[1:-1,:-2]&mask[1:-1,2:])
    return float(1-interior.sum()/mask.sum())
