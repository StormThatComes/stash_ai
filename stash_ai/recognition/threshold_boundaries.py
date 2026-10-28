"""Six-decimal proposal boundaries matching inclusive production gates."""
import math

RESOLUTION = 1000000


def at_or_below(value):
    if round(value, 6) == value:
        return value  # A serialized lattice value is already an inclusive bound.
    result = math.floor(value*RESOLUTION)/RESOLUTION
    if result > value:  # Multiplication may have rounded onto the next integer.
        result = round(result-1/RESOLUTION, 6)
    return result


def strictly_above(value, *, integer=False):
    if integer:
        return math.floor(value)+1
    result = round(at_or_below(value)+1/RESOLUTION, 6)
    if result <= value:
        result = round(result+1/RESOLUTION, 6)
    return result


def at_or_above(value):
    below = at_or_below(value)
    return below if below == value else strictly_above(value)


def sides(value, *, lower=0., upper=1.):
    if value is None or not math.isfinite(value) or not lower <= value <= upper:
        return ()
    return tuple(v for v in (at_or_below(value), strictly_above(value)) if lower <= v <= upper)
