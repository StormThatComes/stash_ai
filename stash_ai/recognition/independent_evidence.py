"""One timestamp or repeated retained crop is one witness within a scan.

Only an explicit pixel fingerprint is deduplicated. Equal embeddings alone do
not imply equal pixels. Legacy evidence retains its timestamp-based semantics.
"""
from .run_cache import metadata


def unit(row):
    if 'quality' in row.keys():
        try:
            record = metadata(row['quality'])
            value = record.get('evidence_pixel_sha256')
            if isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value):
                from .evidence_pixels import canonical_pixel
                return 'pixels', canonical_pixel(value)
            value = record.get('evidence_crop_sha256')
            if isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value):
                return row['scan'], 'pixels', value
        except (ValueError, TypeError, AttributeError):
            pass
    return row['scan'], 'time', row['seconds']


class Votes:
    """Maximum independent timestamp/crop matching, independent of row order.

    A vote consumes both a timestamp and a retained pixel fingerprint. This
    avoids double counting simultaneous crops without collapsing a chain of
    otherwise independent timestamps. Legacy rows have timestamp-local pixels.
    """
    def __init__(self):
        self.edges = {}
        self.units = set()
        self._count = None
        self.owners = {}
        self.unmatched = set()

    def add(self, row):
        stamp = row['scan'], row['seconds']
        fingerprint = unit(row)
        self.units.add(fingerprint)
        if stamp not in self.edges:
            self.unmatched.add(stamp)
        choices = self.edges.setdefault(stamp, set())
        if fingerprint not in choices:
            choices.add(fingerprint)
            self._count = None

    def __contains__(self, value):
        return value in self.units

    def update(self, other):
        self.units.update(other.units)
        for stamp, pixels in other.edges.items():
            if stamp not in self.edges:
                self.unmatched.add(stamp)
            self.edges.setdefault(stamp, set()).update(pixels)
        self._count = None

    def __len__(self):
        if self._count is not None:
            return self._count
        owners = self.owners
        # Iterative augmenting paths avoid Python recursion on long clips.
        for start in sorted(self.unmatched, key=repr):
            pending, parents, visited = [start], {}, {start}
            endpoint = None
            while pending and endpoint is None:
                stamp = pending.pop()
                for pixel in sorted(self.edges[stamp], key=repr):
                    owner = owners.get(pixel)
                    if owner is None:
                        endpoint = stamp, pixel
                        break
                    if owner not in visited:
                        visited.add(owner)
                        parents[owner] = stamp, pixel
                        pending.append(owner)
            if endpoint is not None:
                self.unmatched.remove(start)
                stamp, pixel = endpoint
                owners[pixel] = stamp
                while stamp in parents:
                    previous, pixel = parents[stamp]
                    owners[pixel] = previous
                    stamp = previous
        self._count = len(owners)
        return self._count


def count(rows):
    votes = Votes()
    for row in rows:
        votes.add(row)
    return len(votes)


def representatives(rows):
    """Return canonical proofs for a maximum independent matching."""
    rows = list(rows)
    votes = Votes()
    for row in rows:
        votes.add(row)
    len(votes)
    chosen = {}
    for row in rows:
        pixel = unit(row)
        if votes.owners.get(pixel) == (row['scan'], row['seconds']):
            chosen.setdefault(pixel, row)
    return list(chosen.values())


def grouped_counts(rows, key):
    groups = {}
    for row in rows:
        groups.setdefault(row[key], Votes()).add(row)
    return {name: len(votes) for name, votes in groups.items()}
