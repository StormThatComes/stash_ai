"""Verify compression duplicates before independent evidence is counted.

Perceptual signatures only retrieve candidates. A copy needs matching aspect,
textured RGB pixels (strict residual bounds), and an almost identical embedding.
Aliases always refer to the original representative, never a transitive chain.
Only the private replay's compact metadata is changed.
"""
import json
import numpy as np
from .run_cache import metadata
from .progress import tracking


def signature(pixels):
    gray = pixels.reshape(32, 32, 3).astype('float64').mean(axis=2)
    coarse = gray.reshape(8, 4, 8, 4).mean(axis=(1, 3))
    return np.packbits(coarse[:, 1:] > coarse[:, :-1]).tobytes()


def verified(left, right, a, b, aspect_a, aspect_b):
    if not (isinstance(aspect_a, (int, float)) and isinstance(aspect_b, (int, float))):
        return False
    if not (0 < aspect_a and 0 < aspect_b) or abs(aspect_a / aspect_b - 1) > .005:
        return False
    residual = left.astype('float64') - right.astype('float64')
    return (min(float(left.std()), float(right.std())) >= 12 and
            float(np.max(np.abs(residual))) <= 8 and float(np.mean(residual * residual)) <= 4 and
            a.shape == b.shape and a.size > 0 and float(np.dot(a, b)) >= .999)


def canonicalize(db):
    db.execute('CREATE TEMP TABLE pixel_witnesses(signature BLOB, fingerprint TEXT PRIMARY KEY, pixels BLOB, embedding BLOB, aspect REAL)')
    db.execute('CREATE INDEX pixel_witness_signature ON pixel_witnesses(signature)')
    try:
        rows = db.execute("SELECT id,quality,embedding FROM observations WHERE rejected IS NULL "
            "AND json_extract(quality,'$.evidence_thumbnail') IS NOT NULL "
            "ORDER BY json_extract(quality,'$.evidence_pixel_sha256'),id")
        with tracking(rows, 'Verify duplicate VTT crops', unit='observation') as tracked:
            for row in tracked:
                record = metadata(row['quality'])
                raw = bytes.fromhex(record['evidence_thumbnail'])
                fingerprint = record.get('evidence_pixel_sha256')
                if len(raw) != 3072 or not fingerprint:
                    continue
                pixels = np.frombuffer(raw, dtype='uint8')
                sample = np.frombuffer(row['embedding'], dtype='<f4')
                bucket = signature(pixels)
                alias = None
                for candidate in db.execute('SELECT * FROM pixel_witnesses WHERE signature=? ORDER BY fingerprint', (bucket,)):
                    if fingerprint == candidate['fingerprint'] or verified(pixels,
                            np.frombuffer(candidate['pixels'], dtype='uint8'), sample,
                            np.frombuffer(candidate['embedding'], dtype='<f4'),
                            record.get('evidence_aspect'), candidate['aspect']):
                        alias = candidate['fingerprint']
                        break
                if alias is None:
                    db.execute('INSERT OR IGNORE INTO pixel_witnesses VALUES (?,?,?,?,?)',
                        (bucket, fingerprint, raw, row['embedding'], record.get('evidence_aspect')))
                elif alias != fingerprint:
                    record['evidence_pixel_sha256'] = alias
                    db.execute('UPDATE observations SET quality=? WHERE id=?',
                        (json.dumps(record, separators=(',', ':')), row['id']))
    finally:
        db.execute('DROP TABLE pixel_witnesses')
