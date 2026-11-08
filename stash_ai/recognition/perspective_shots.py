"""Scene-local shot boundaries from retained, low-frequency VTT pixels."""
import numpy as np


def measurements(previous, current):
    residual = current - previous
    difference = np.abs(residual).mean()
    structure = np.abs(residual - residual.mean(axis=0)).mean()
    colour = np.ptp(residual.mean(axis=0))
    return difference,structure,colour


def camera_shift(previous,current):
    """A textured 4x4 thumbnail can certify a one-cell pan in a single axis."""
    a,b = previous.reshape(4,4,3),current.reshape(4,4,3)
    if min(a.std(axis=(0,1)).mean(),b.std(axis=(0,1)).mean())<12:
        return None
    trials = []
    for dx,dy in ((0,0),(-1,0),(1,0),(0,-1),(0,1)):
        first = a[max(0,-dy):4-max(dy,0),max(0,-dx):4-max(dx,0)]
        second = b[max(0,dy):4-max(-dy,0),max(0,dx):4-max(-dx,0)]
        trials.append((float(np.abs(first-second).mean()),dx,dy))
    (best,dx,dy),(runner,_,_) = sorted(trials)[:2]
    if (dx or dy) and best<=6 and runner-best>=12:
        return dx,dy
    return None


def change(previous, current, adjacent):
    difference,structure,colour = measurements(previous,current)
    if structure > 16 or colour > 30:
        return 'continuous' if adjacent and camera_shift(previous,current) else 'cut'
    return 'continuous' if adjacent and difference <= 20 else 'uncertain'


def advance(image, cue, previous):
    from .body_perspectives import frame_signature
    signature = frame_signature(image)
    shot = 0 if cue == 0 else None
    if previous and previous[0] == cue-1:
        before = np.frombuffer(bytes.fromhex(previous[2]),dtype='uint8').astype(float).reshape(-1,3)
        current = np.frombuffer(bytes.fromhex(signature),dtype='uint8').astype(float).reshape(-1,3)
        shot = cue if change(before,current,True)=='cut' else previous[1]
    return cue,shot,signature


def boundaries(by_cue, geometry):
    shot = 0
    shots, boundary = {}, {}
    previous = None
    for cue in sorted(by_cue):
        signature = geometry[by_cue[cue][0]['id']][2].reshape(-1, 3)
        kind = 'start'
        if previous is not None:
            before, earlier = previous
            kind = change(earlier,signature,cue==before+1)
            if kind == 'cut':
                shot += 1
        captured = geometry[by_cue[cue][0]['id']][0].get('view_shot')
        shots[cue], boundary[cue] = (('captured',captured) if captured is not None else ('retained',shot)), kind
        previous = cue, signature
    return shots, boundary
