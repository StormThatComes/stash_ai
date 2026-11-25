"""Edge-spread covariance proxy, shared by NumPy and Torch.

Only isolated, monotonic edges with four supported normal directions participate.
Fit a common isotropic spread plus a directional spread. The result is a blur
proxy, not a lens diagnosis; irregular motion and structured textures can remain
ambiguous. All samples stay inside the existing 15x15 foreground footprint.
"""
import math


def estimate(smooth, support, noise, xp):
    from .blur_quality import _pool, _quartile
    # Smooth already excludes two source pixels per side. Centers retain the
    # original radius-seven footprint; profiles extend at most five more pixels.
    height, width = smooth.shape[-2:]
    def shifted(dy, dx):
        return smooth[:, 5+dy:height-5+dy, 5+dx:width-5+dx]
    gx = (shifted(0, 1)-shifted(0, -1))*.5
    gy = (shifted(1, 0)-shifted(-1, 0))*.5
    magnitude = (gx*gx+gy*gy)**.5
    spreads, counts, tiles = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        step = math.hypot(dx, dy)
        derivative = (gx*dx+gy*dy)/step
        # Diverse edge normals prevent sharp stripes from looking like motion.
        selected = support & (abs(derivative) >= .924*magnitude) & (magnitude > 1.)
        profile = [shifted(dy*k, dx*k) for k in range(-5, 6)]
        jumps = [profile[i+1]-profile[i] for i in range(10)]
        total = abs(profile[-1]-profile[0])
        variation = sum(abs(jump) for jump in jumps)
        peak = abs(profile[6]-profile[4])/(2*step)
        # A monotonic edge must peak at the center. Reject texture oscillations,
        # compression ringing, and displaced responses from nearby boundaries.
        selected = selected & (variation <= total*1.1) & (total > 8.) & (peak > 1.)
        for i in range(1, 10):
            selected = selected & (abs(profile[i+1]-profile[i-1])/(2*step) <= peak*1.05)
        # Equivalent Gaussian edge width from contrast / central derivative.
        variance = (total/peak.clip(.01, None))**2/(2*math.pi)
        # Local noise is measured on the same tiles by the legacy estimator.
        weights = selected*1.
        n = _pool(weights)
        signal = _pool(peak*weights)/n.clip(1, None)
        tile_valid = (n >= 2) & (signal > 3*noise+.5)
        local = _pool(variance*weights)/n.clip(1, None)
        spreads.append(_quartile(local, tile_valid, xp))
        counts.append((n*tile_valid).sum(axis=1))
        tiles.append(tile_valid.sum(axis=1))
    values = xp.stack(spreads, axis=1)
    finite = xp.isfinite(values).all(axis=1)
    values = xp.where(xp.isfinite(values), values, 0.)
    xx, yy, plus, minus = (values[:, i] for i in range(4))
    mean = (xx+yy)*.5
    cross = (plus-minus)*.5
    radius = (((xx-yy)*.5)**2+cross**2)**.5
    minor, major = mean-radius, mean+radius
    residual = abs((plus+minus)*.5-mean)/mean.clip(.01, None)
    count = xp.stack(counts, axis=1)
    tile = xp.stack(tiles, axis=1)
    enough = (count >= 8).all(axis=1) & (tile >= 2).all(axis=1)
    good = finite & enough & (minor >= .5) & (residual < .3)
    # Base binomial smoothing contributes variance 1; finite derivatives add
    # approximately 1/3. The two proxies intentionally have independent scales.
    focus = 100/(1+(minor-4/3).clip(0, None)/3)
    motion = 100/(1+(major-minor).clip(0, None)/6)
    confidence = xp.where(good, (1-residual).clip(0, 1), 0.)
    return xp.stack((xp.where(good, focus, 0.), xp.where(good, motion, 0.), confidence), axis=1)
