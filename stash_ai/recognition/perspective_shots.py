"""Scene-local shot boundaries from retained, low-frequency VTT pixels."""
import numpy as np


def boundaries(by_cue, geometry):
    shot = 0
    shots, boundary = {}, {}
    previous = None
    for cue in sorted(by_cue):
        signature = geometry[by_cue[cue][0]['id']][2].reshape(-1, 3)
        change = 'start'
        if previous is not None:
            before, earlier = previous
            residual = signature - earlier
            difference = np.abs(residual).mean()
            # Discount uniform exposure shifts when certifying a different shot.
            structure = np.abs(residual - residual.mean(axis=0)).mean()
            colour = np.ptp(residual.mean(axis=0))
            change = ('cut' if structure > 16 or colour > 30 else
                      'uncertain' if difference > 20 or cue != before+1 else 'continuous')
            if change == 'cut':
                shot += 1
        shots[cue], boundary[cue] = shot, change
        previous = cue, signature
    return shots, boundary
