"""Public memory limits use MB (1 MB = 1024**2 bytes, i.e. MiB)."""
import argparse
import math

BYTES_PER_MB = 1024**2


def mb_to_bytes(value, *, name='Memory limit', minimum=0, maximum=None):
    """Validate numeric MB before truncating any sub-byte remainder."""
    valid = type(value) in (int, float)
    try:
        valid = valid and math.isfinite(value) and value >= minimum
        valid = valid and (maximum is None or value <= maximum)
        scaled = value * BYTES_PER_MB if valid else None
        valid = valid and math.isfinite(scaled)
    except OverflowError:
        valid = False
    if not valid:
        bounds = f'>= {minimum}' if maximum is None else f'in [{minimum}, {maximum}]'
        raise ValueError(f'{name}: expected a finite number {bounds} MB (1 MB = 1048576 bytes).')
    result = int(scaled)
    if value > 0 and result == 0:
        raise ValueError(f'{name}: a positive MB limit must represent at least one byte.')
    return result


def bytes_to_mb(value):
    """Keep fractional MB when exporting an integer byte budget."""
    result = value / BYTES_PER_MB
    return int(result) if result.is_integer() else result


def mb_argument(value):
    """Argparse numeric MB input, including fractions and scientific notation."""
    try:
        number = float(value)
        mb_to_bytes(number)
        return number
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
