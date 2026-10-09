"""Compact checkpoints and atomic completion of retained inference results."""
import zlib

from stash_ai.storage.codec import compress, encode, decode


def clear_checkpoint(store, scene, model):
    from stash_ai.storage.preview import delete
    for table in ("checkpoints", "checkpoint_frames", "checkpoint_analysis"):
        delete(store, table, scene, model)
        store.connection.execute(f"DELETE FROM {table} WHERE scope=? AND scene=? AND model=?",
                                 (store.scope, str(scene), model))


def finish(store, scene, model, outputs):
    # Results and checkpoint removal commit together, including zero detections.
    with store.connection:
        store.results.put_in_transaction(scene, model, outputs)
        clear_checkpoint(store, scene, model)
        for kind in outputs:
            from stash_ai.storage.preview import delete
            delete(store, 'refresh_requests', scene, model, kind)
            store.connection.execute("DELETE FROM refresh_requests WHERE scope=? AND scene=? AND model=? AND kind=?",
                                     (store.scope, str(scene), model, kind))


class Checkpoint:
    def __init__(self, store, scene, model, signature, total, labels, every=8, *, keep=None):
        self.store, self.scene, self.model = store, str(scene), model
        self.signature, self.total, self.labels, self.every = signature, total, labels, every
        self.pending, self.batches, self.processed, self.top_scores = [], 0, 0, None
        self.saved_processed = 0
        self.resumed_frames = 0
        self.analysis = None
        self.keep = keep
        self.aggregation_verified = False
        self.restore_reason, self.previous_signature, self.on_restore = 'missing', None, None

    def _notify_restore(self):
        if self.on_restore is not None:
            self.on_restore(self)

    def restore(self):
        import numpy as np
        row = self.store.connection.execute(
            "SELECT signature,processed,total,top_rows,top_scores FROM " + self.store.table('checkpoints') + " WHERE scope=? AND scene=? AND model=?",
            (self.store.scope, self.scene, self.model)).fetchone()
        if not row or row[0] != self.signature or row[2] != self.total:
            self.previous_signature = row[0] if row else None
            self.restore_reason = ('missing' if not row else 'signature_changed' if row[0] != self.signature else 'total_changed')
            with self.store.connection:
                clear_checkpoint(self.store, self.scene, self.model)
            self._notify_restore()
            return 0, np.empty((0, self.labels), np.float32), []
        _, processed, _, rows, blob = row
        self.previous_signature, self.restore_reason = row[0], 'compatible'
        if not 0 <= processed <= self.total or not 0 <= rows <= processed:
            raise ValueError("Invalid checkpoint dimensions.")
        from stash_ai.storage.checkpoint_contract import registered_contract, retained_rows, validate_chunks
        contract = registered_contract(self.store.connection, self.store.scope, self.scene,
                                       self.model, self.signature,
                                       table=self.store.table('inference_identities'))
        required = (min(self.keep, processed) if self.keep is not None
                    else retained_rows(contract, processed) if contract is not None else None)
        if required is not None and rows != required:
            raise ValueError('Incomplete checkpoint aggregate scores; use --restart.')
        self.aggregation_verified = required is not None
        top = np.frombuffer(zlib.decompress(blob), dtype="<f4").reshape(rows, self.labels).copy()
        if not np.isfinite(top).all() or ((top < 0) | (top > 1)).any():
            raise ValueError("Invalid checkpoint scores.")
        chunks = [(end, decode(data)) for end, data in self.store.connection.execute(
                "SELECT processed,data FROM " + self.store.table('checkpoint_frames') + " WHERE scope=? AND scene=? AND model=? ORDER BY processed",
                (self.store.scope, self.scene, self.model))]
        frames = validate_chunks(chunks, processed)
        self.processed = self.saved_processed = processed
        self.resumed_frames = processed
        self.top_scores = top
        state = self.store.connection.execute(
            "SELECT data FROM " + self.store.table('checkpoint_analysis') + " WHERE scope=? AND scene=? AND model=?",
            (self.store.scope, self.scene, self.model)).fetchone()
        self.analysis = decode(state[0]) if state else None
        if self.analysis is not None:
            from stash_ai.evidence.temporal.statistics import validate_checkpoint
            validate_checkpoint(self.analysis, processed=processed, labels=self.labels)
        from stash_ai.db.payloads import valid_frames
        if not valid_frames(frames, checkpoint=True):
            raise ValueError('Invalid checkpoint frame evidence; use --restart.')
        self._notify_restore()
        return processed, top, frames

    def record(self, processed, top_scores, frames, analysis=None, *, scene_frames=None):
        self.processed, self.top_scores = processed, top_scores
        rows = [dict(start=f.cue.start, end=f.cue.end, tags=list(f.tags)) for f in frames]
        if scene_frames is not None:
            if not rows:
                rows = [dict(start=f['start'], end=f['end'], tags=[]) for f in scene_frames]
            if len(rows) != len(scene_frames):
                raise ValueError('Scene/marker checkpoint timelines do not match.')
            for row, scene in zip(rows, scene_frames):
                row['scene_tags'] = scene['tags']
        self.pending.extend(rows)
        self.analysis = analysis
        self.batches += 1
        if self.batches >= self.every:
            self.flush()

    def flush(self):
        from stash_ai.runtime.timing import stage
        with stage(getattr(self.store, 'report', None), 'checkpoint_write'):
            self._flush()

    def _flush(self):
        if self.top_scores is None or self.processed == self.saved_processed:
            return
        from stash_ai.runtime.timing import stage
        report = getattr(self.store, 'report', None)
        # Prepare immutable blobs before opening the write transaction. A
        # failed encode cannot modify SQLite or advance the durable prefix.
        with stage(report, 'checkpoint_prepare'):
            scores = compress(self.top_scores.astype("<f4", copy=False).tobytes(), report,
                              stage_name='checkpoint_compress')
            frames = encode(self.pending, report, stage_prefix='checkpoint') if self.pending else None
            analysis = (encode(self.analysis, report, stage_prefix='checkpoint')
                        if self.analysis is not None else None)
        db = self.store.connection
        with stage(report, 'checkpoint_sqlite_write'):
            with db:
                db.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(scope,scene,model) DO UPDATE SET signature=excluded.signature,processed=excluded.processed,"
                    "total=excluded.total,top_rows=excluded.top_rows,top_scores=excluded.top_scores",
                    (self.store.scope, self.scene, self.model, self.signature, self.processed, self.total,
                     len(self.top_scores), scores))
                if frames is not None:
                    db.execute("INSERT INTO checkpoint_frames VALUES (?,?,?,?,?)",
                               (self.store.scope, self.scene, self.model, self.processed, frames))
                if analysis is not None:
                    db.execute("INSERT INTO checkpoint_analysis VALUES (?,?,?,?) ON CONFLICT(scope,scene,model) "
                               "DO UPDATE SET data=excluded.data", (self.store.scope, self.scene, self.model, analysis))
        self.saved_processed = self.processed
        self.pending, self.batches = [], 0
