"""Validate checkpoint dimensions, frame progress and float32 scores."""
from __future__ import annotations

import struct
import zlib

from ..common import count_rows
from .common import tick


def check_checkpoints(db, tables, errors, progress=None, warnings=None):
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

        from stash_ai.storage.checkpoint_contract import (
            registered_contract, retained_rows, validate_chunk_prefix, MalformedCheckpointChunk)
        from stash_ai.storage.codec import decode
        for scope, scene, model, signature, processed, top_rows in db.execute(
                'SELECT scope,scene,model,signature,processed,top_rows FROM checkpoints'):
            contract = (registered_contract(db, scope, scene, model, signature)
                        if 'inference_identities' in tables else None)
            if contract is not None:
                try:
                    if top_rows != retained_rows(contract, processed):
                        errors['checkpoint_aggregate_incomplete'] += 1
                except (KeyError, TypeError, ValueError):
                    errors['checkpoint_aggregate_incomplete'] += 1
            elif warnings is not None:
                warnings['checkpoint_aggregation_unverified'] += 1
            has_chunks = ('checkpoint_frames' in tables and bool(db.execute(
                'SELECT 1 FROM checkpoint_frames WHERE scope=? AND scene=? AND model=? LIMIT 1',
                (scope, scene, model)).fetchone()))
            # Old checkpoints without provenance cannot prove capture policy.
            # A known policy, analysis or any chunks must form a complete prefix.
            analysis = ('checkpoint_analysis' in tables and db.execute(
                'SELECT 1 FROM checkpoint_analysis WHERE scope=? AND scene=? AND model=?',
                (scope, scene, model)).fetchone())
            if contract is not None or has_chunks or analysis:
                def decoded_chunks():
                    if 'checkpoint_frames' not in tables:
                        return
                    for end, blob in db.execute(
                            'SELECT processed,data FROM checkpoint_frames '
                            'WHERE scope=? AND scene=? AND model=? ORDER BY processed',
                            (scope, scene, model)):
                        yield end, decode(blob)
                try:
                    validate_chunk_prefix(decoded_chunks(), processed)
                except MalformedCheckpointChunk:
                    continue  # Already counted by state_payload_invalid.
                except (KeyError, TypeError, zlib.error):
                    continue  # Already counted by state_payload_invalid.
                except ValueError:
                    errors['checkpoint_frame_incomplete'] += 1

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

        if 'checkpoints' in tables:
            from stash_ai.storage.codec import decode
            from stash_ai.evidence.temporal.statistics import validate_checkpoint
            for blob, processed, top_rows, scores in db.execute(
                    'SELECT a.data,c.processed,c.top_rows,c.top_scores FROM checkpoint_analysis AS a '
                    'JOIN checkpoints AS c USING(scope,scene,model)'):
                try:
                    raw = zlib.decompress(scores)
                    labels = len(raw) // (top_rows * 4) if top_rows > 0 else None
                    validate_checkpoint(decode(blob), processed=processed, labels=labels)
                except (KeyError, TypeError, ValueError, zlib.error):
                    errors['checkpoint_analysis_progress'] += 1
