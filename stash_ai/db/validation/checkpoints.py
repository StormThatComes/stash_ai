"""Validate checkpoint dimensions, frame progress and float32 scores."""
from __future__ import annotations

import struct
import zlib

from ..common import count_rows
from .common import tick


def check_checkpoints(db, tables, errors, progress=None):
    if "checkpoints" in tables:
        errors["checkpoint_dimensions"] += db.execute("""
            SELECT COUNT(*) FROM checkpoints
            WHERE processed <= 0 OR total <= 0 OR processed > total
               OR top_rows < 0 OR top_rows > processed
               OR typeof(processed) != 'integer' OR typeof(total) != 'integer'
               OR typeof(top_rows) != 'integer'
               OR signature = ''
        """).fetchone()[0]

        for top_rows, blob in db.execute("SELECT top_rows,top_scores FROM checkpoints"):
            tick(progress)
            try:
                raw = zlib.decompress(blob)
                valid = type(top_rows) is int and (
                    (top_rows == 0 and len(raw) == 0)
                    or (top_rows > 0 and len(raw) > 0 and len(raw) % (top_rows * 4) == 0)
                )
                # The shape alone accepts NaN/Inf and out-of-range scores. Match
                # inference's probability contract without importing NumPy.
                valid = valid and all(0 <= score <= 1 for score, in struct.iter_unpack("<f", raw))
                if not valid:
                    errors["checkpoint_scores"] += 1
            except (TypeError, zlib.error):
                errors["checkpoint_scores"] += 1

    if "checkpoint_frames" in tables:
        if "checkpoints" in tables:
            errors["checkpoint_orphan_frames"] += db.execute("""
                SELECT COUNT(*) FROM checkpoint_frames AS f
                WHERE NOT EXISTS (
                    SELECT 1 FROM checkpoints AS c
                    WHERE c.scope=f.scope AND c.scene=f.scene AND c.model=f.model
                )
            """).fetchone()[0]
            errors["checkpoint_frame_progress"] += db.execute("""
                SELECT COUNT(*) FROM checkpoint_frames AS f
                JOIN checkpoints AS c
                  ON c.scope=f.scope AND c.scene=f.scene AND c.model=f.model
                WHERE f.processed <= 0 OR f.processed > c.processed
            """).fetchone()[0]
        else:
            errors["checkpoint_orphan_frames"] += count_rows(db, tables, "checkpoint_frames")

    if "checkpoint_analysis" in tables:
        if "checkpoints" in tables:
            errors["checkpoint_orphan_analysis"] += db.execute("""
                SELECT COUNT(*) FROM checkpoint_analysis AS a
                WHERE NOT EXISTS (
                    SELECT 1 FROM checkpoints AS c
                    WHERE c.scope=a.scope AND c.scene=a.scene AND c.model=a.model
                )
            """).fetchone()[0]
        else:
            errors["checkpoint_orphan_analysis"] += count_rows(db, tables, "checkpoint_analysis")
