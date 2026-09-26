"""Compact progress/checkpoints. No image files or full frame-score matrices."""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import zlib

from stash_state_codec import encode, decode
from stash_taggers import metadata_files, weight_files

PREDICTION_FORMAT = 2


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def initialize_progress(store, previous_version):
    db = store.connection
    db.executescript("""
        CREATE TABLE IF NOT EXISTS result_index (
            scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL, kind TEXT NOT NULL,
            inference INTEGER NOT NULL, populated INTEGER NOT NULL, signature TEXT,
            PRIMARY KEY(scope,scene,model,kind));
        CREATE TABLE IF NOT EXISTS refresh_requests (
            scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL, kind TEXT NOT NULL,
            PRIMARY KEY(scope,scene,model,kind));
        CREATE TABLE IF NOT EXISTS checkpoints (
            scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
            signature TEXT NOT NULL, processed INTEGER NOT NULL, total INTEGER NOT NULL,
            top_rows INTEGER NOT NULL, top_scores BLOB NOT NULL,
            PRIMARY KEY(scope,scene,model));
        CREATE TABLE IF NOT EXISTS checkpoint_frames (
            scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
            processed INTEGER NOT NULL, data BLOB NOT NULL,
            PRIMARY KEY(scope,scene,model,processed));
        CREATE TABLE IF NOT EXISTS checkpoint_analysis (
            scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL, data BLOB NOT NULL,
            PRIMARY KEY(scope,scene,model));
    """)
    if previous_version < 2:
        # One-time migration; future status checks never decompress prediction blobs.
        with db:
            for scope, scene, model, kind, data in db.execute("SELECT scope,scene,model,kind,data FROM results"):
                index_result(store, scene, model, kind, decode(data), scope=scope)
    if previous_version < 4:
        with db:
            # Video sampling is retired; base VTT results and checkpoints stay valid.
            db.execute("DROP TABLE IF EXISTS refinement_progress")
            db.execute("PRAGMA user_version=4")


def index_result(store, scene, model, kind, segments, scope=None):
    from stash_thresholds import indexed_signature
    inference = next((segment for segment in reversed(segments)
                      if segment["meta"].get("origin") == "inference"), None)
    store.connection.execute("INSERT INTO result_index VALUES (?,?,?,?,?,?,?) "
        "ON CONFLICT(scope,scene,model,kind) DO UPDATE SET inference=excluded.inference,"
        "populated=excluded.populated,signature=excluded.signature",
        (scope or store.scope, str(scene), model, kind,
         int(inference is not None),
         int(any(s["items"] for s in segments)),
         indexed_signature(inference) if inference is not None else None))


def output_kinds(app):
    return [kind for kind, enabled in (("scene", app.APPLY_SCENE_TAGS), ("markers", app.CREATE_MARKERS)) if enabled]


def completed(store, scene, model, kinds, signature=None):
    cache_key = ('status', str(scene), model)
    rows = store.cached(cache_key)
    if rows is None:
        rows = {kind: (inference, populated, saved) for kind, inference, populated, saved in store.connection.execute(
        "SELECT kind,inference,populated,signature FROM " + store.table('result_index') + " AS i WHERE scope=? AND scene=? AND model=? "
        "AND NOT EXISTS (SELECT 1 FROM " + store.table('refresh_requests') + " AS r WHERE "
        "r.scope=i.scope AND r.scene=i.scene AND r.model=i.model AND r.kind=i.kind)",
        (store.scope, str(scene), model))}
        store.retain(cache_key, rows)
    return all(kind in rows and
               ((rows[kind][0] and rows[kind][2] == signature) if signature is not None
                else (rows[kind][0] or rows[kind][1])) for kind in kinds)


def reuse_replacement_policy(store, scene, model, kinds, signature, replace):
    """Reuse current inference while applying replacement semantics to retained history."""
    from stash_thresholds import indexed_signature
    matches = {}
    records = {}
    for kind in kinds:
        if store.connection.execute(
                "SELECT 1 FROM " + store.table('refresh_requests') +
                " WHERE scope=? AND scene=? AND model=? AND kind=?",
                (store.scope, str(scene), model, kind)).fetchone():
            return None
        segments = store.records(scene, kind, model, borrow=True).get(model, [])
        # Match the effective policy as completed() does. A raw inference
        # signature can match restored thresholds while overrides still retain
        # the raised values; let threshold reuse reconcile those first.
        current = next((segment for segment in reversed(segments)
                        if segment['meta'].get('origin') == 'inference'
                        and indexed_signature(segment) == signature), None)
        if current is None:
            return None
        records[kind] = segments
        matches[kind] = current

    targets = [(kind, matches[kind]) for kind in kinds
               if replace.get(kind, True)
               and (len(records[kind]) != 1 or records[kind][0] != matches[kind])]
    if not targets:
        return 0
    # Never rewrite source evidence underneath an interrupted publication plan.
    if any(store.snapshot(scene, kind, borrow=True)[1] is not None for kind in ('scene', 'markers')):
        raise RuntimeError(f'{scene}: finish the pending merge or use --abandon-pending before changing replacement policy.')
    with store.connection:
        for kind, segment in targets:
            store.replace_records(scene, model, kind, [segment])
    return len(targets)

def collapse_replacement_history(store, scene, model, kinds, replace):
    """Apply replacement after another cache path has already proven the latest result."""
    targets = []
    for kind in kinds:
        if not replace.get(kind, True):
            continue
        segments = store.records(scene, kind, model, borrow=True).get(model, [])
        if len(segments) <= 1:
            continue
        segment = segments[-1]
        if segment['meta'].get('origin') != 'inference':
            return None
        targets.append((kind, segment))
    if not targets:
        return 0
    if any(store.snapshot(scene, kind, borrow=True)[1] is not None for kind in ('scene', 'markers')):
        raise RuntimeError(f'{scene}: finish the pending merge or use --abandon-pending before changing replacement policy.')
    with store.connection:
        for kind, segment in targets:
            store.replace_records(scene, model, kind, [segment])
    return len(targets)


def clear_checkpoint(store, scene, model):
    from stash_preview import delete
    for table in ("checkpoints", "checkpoint_frames", "checkpoint_analysis"):
        delete(store, table, scene, model)
        store.connection.execute(f"DELETE FROM {table} WHERE scope=? AND scene=? AND model=?",
                                 (store.scope, str(scene), model))


def request_refresh(store, scenes, models, kinds):
    with store.connection:
        for scene in scenes:
            for model in models:
                store.cache.discard(('status', str(scene), model))
                clear_checkpoint(store, scene, model)
                for kind in kinds:
                    store.connection.execute("INSERT OR IGNORE INTO refresh_requests VALUES (?,?,?,?)",
                                             (store.scope, str(scene), model, kind))


def finish(store, scene, model, outputs, replace):
    # Results and checkpoint removal commit together, including zero detections.
    with store.connection:
        store._put(scene, model, outputs, replace)
        clear_checkpoint(store, scene, model)
        for kind in outputs:
            from stash_preview import delete
            delete(store, 'refresh_requests', scene, model, kind)
            store.connection.execute("DELETE FROM refresh_requests WHERE scope=? AND scene=? AND model=? AND kind=?",
                                     (store.scope, str(scene), model, kind))


def preload_status(store):
    """One small ordered query instead of per-pair status reads across all passes."""
    if not store.cache.limit:
        return
    last, values = None, {}
    rows = store.connection.execute(
        "SELECT i.scene,i.model,i.kind,i.inference,i.populated,i.signature FROM " + store.table('result_index') + " AS i "
        "LEFT JOIN " + store.table('refresh_requests') + " AS r ON r.scope=i.scope AND r.scene=i.scene AND r.model=i.model AND r.kind=i.kind "
        "WHERE i.scope=? AND r.kind IS NULL ORDER BY i.scene,i.model", (store.scope,))
    for scene, model, kind, inference, populated, signature in rows:
        key = ('status', scene, model)
        if key != last:
            if last is not None:
                store.cache.put(last, values)
            last, values = key, {}
        values[kind] = (inference, populated, signature)
    if last is not None:
        store.cache.put(last, values)


def model_identity(app, config, vocabulary):
    def stamp(path):
        path = Path(path).resolve()
        try:
            stat = path.stat()
            return [str(path), stat.st_size, stat.st_mtime_ns]
        except FileNotFoundError:
            return [str(path), None, None]
    files = [stamp(config.model_dir / name) for name in (*metadata_files(config.source), *weight_files(config))]
    loader = {}
    if config.source.key == "joytag":
        from stash_joytag import LOADER_REVISION, LOADER_SHA256
        loader = {"loader": {"revision": LOADER_REVISION, "sha256": LOADER_SHA256}}
    return {"format": PREDICTION_FORMAT, "source": config.source.key, "repo": config.source.repo,
            "revision": config.source.revision, "files": files, **loader,
            # Checkpoint score columns depend on label order, not just the set.
            "vocabulary": digest(vocabulary.labels), "batch_size": config.batch_size,
            "scene_threshold": config.scene_threshold, "marker_threshold": config.marker_threshold,
            "aggregation": app.AGGREGATION, "top_k": app.TOP_K_FRAMES,
            "retention": {"temporal": app.TEMPORAL_POLICY.enabled,
                          "marker_ratio": app.TEMPORAL_POLICY.marker_continue_ratio if app.TEMPORAL_POLICY.enabled else 1.0,
                          "scene_ratio": app.TEMPORAL_POLICY.scene_support_ratio if app.TEMPORAL_POLICY.enabled else 1.0},
            "frame_blocklist": sorted(app.MARKER_TAG_BLOCKLIST)}


class VerifiedMedia:
    """Hash input bytes; retain at most two compressed sheets within a RAM limit.

    Never writes media to disk. A second fetch during inference verifies the
    preflight hash so a regenerated sprite cannot silently reuse a checkpoint.
    """
    def __init__(self, stash, limit=16 * 1024 * 1024, expected=None):
        self.stash, self.limit, self.size = stash, limit, 0
        self.cache, self.hashes = OrderedDict(), dict(expected or {})

    def get_media(self, url):
        if url in self.cache:
            self.cache.move_to_end(url)
            return self.cache[url]
        data = self.stash.get_media(url)
        checksum = hashlib.sha256(data).hexdigest()
        if url in self.hashes and checksum != self.hashes[url]:
            raise RuntimeError("Input media changed during this scene; rerun to use the new input.")
        self.hashes[url] = checksum
        if len(data) <= self.limit:
            while self.cache and (len(self.cache) >= 2 or self.size + len(data) > self.limit):
                self.size -= len(self.cache.popitem(last=False)[1])
            self.cache[url] = data
            self.size += len(data)
        return data


def inspect_input(app, stash, paths, identity):
    media = VerifiedMedia(stash)
    cues, total = app.fetch_cues(media, paths)
    if not cues:
        raise ValueError("No VTT cues to analyze.")
    for url in dict.fromkeys(cue.image for cue in cues):
        media.get_media(url)
    cue_identity = [[cue.start, cue.end, cue.image, list(cue.rect)] for cue in cues]
    signature = digest({"model": identity, "cues": cue_identity, "media": media.hashes})
    return cues, media, signature


class RunMedia:
    """One verified media snapshot for sequential model passes, RAM only."""
    def __init__(self, stash, cache, report=None):
        self.stash, self.cache, self.report = stash, cache, report
        self.token, self.hashes = object(), {}

    def get_media(self, url):
        key = ('media', self.token, url)
        saved = self.cache.get(key)
        if saved is not None:
            if self.report is not None:
                self.report.counts['media_cache_hits'] += 1
            return saved
        data = self.stash.get_media(url)
        checksum = hashlib.sha256(data).hexdigest()
        if url in self.hashes and checksum != self.hashes[url]:
            raise RuntimeError('Stash media changed during model passes; rerun to use one consistent input.')
        self.hashes[url] = checksum
        self.cache.put(key, data)
        if self.report is not None:
            self.report.counts['media_downloads'] += 1
            self.report.counts['media_download_bytes'] += len(data)
        return data


class RunInputs:
    """Share verified cue manifests without changing any inference signatures."""
    def __init__(self, stash, cache, report=None):
        self.cache, self.report = cache, report
        self.media = RunMedia(stash, cache, report)
        self.token, self.identities = object(), {}

    def inspect(self, app, paths, model_identity):
        key = digest({'paths': paths, 'sample_limit': app.MAX_FRAMES_PER_SCENE})
        cached = self.cache.get(('input', self.token, key))
        if cached is None:
            cues, media, signature = inspect_input(app, self.media, paths, model_identity)
            cue_identity = [[c.start, c.end, c.image, list(c.rect)] for c in cues]
            frozen = digest({'cues': cue_identity, 'media': media.hashes})
            if key in self.identities and frozen != self.identities[key]:
                raise RuntimeError('Stash VTT changed during model passes; rerun the analysis.')
            self.identities[key] = frozen
            self.cache.put(('input', self.token, key), (cues, dict(media.hashes), cue_identity))
            return cues, media, signature
        cues, hashes, cue_identity = cached
        media = VerifiedMedia(self.media, limit=0, expected=hashes)
        signature = digest({'model': model_identity, 'cues': cue_identity, 'media': hashes})
        if self.report is not None:
            self.report.counts['input_manifest_hits'] += 1
        return cues, media, signature


class Checkpoint:
    def __init__(self, store, scene, model, signature, total, labels, every=8):
        self.store, self.scene, self.model = store, str(scene), model
        self.signature, self.total, self.labels, self.every = signature, total, labels, every
        self.pending, self.batches, self.processed, self.top_scores = [], 0, 0, None
        self.saved_processed = 0
        self.analysis = None

    def restore(self):
        import numpy as np
        row = self.store.connection.execute(
            "SELECT signature,processed,total,top_rows,top_scores FROM " + self.store.table('checkpoints') + " WHERE scope=? AND scene=? AND model=?",
            (self.store.scope, self.scene, self.model)).fetchone()
        if not row or row[0] != self.signature or row[2] != self.total:
            with self.store.connection:
                clear_checkpoint(self.store, self.scene, self.model)
            return 0, np.empty((0, self.labels), np.float32), []
        _, processed, _, rows, blob = row
        if not 0 <= processed <= self.total or not 0 <= rows <= processed:
            raise ValueError("Invalid checkpoint dimensions.")
        top = np.frombuffer(zlib.decompress(blob), dtype="<f4").reshape(rows, self.labels).copy()
        if not np.isfinite(top).all():
            raise ValueError("Invalid checkpoint scores.")
        frames = []
        for data, in self.store.connection.execute(
                "SELECT data FROM " + self.store.table('checkpoint_frames') + " WHERE scope=? AND scene=? AND model=? ORDER BY processed",
                (self.store.scope, self.scene, self.model)):
            frames.extend(decode(data))
        self.processed = self.saved_processed = processed
        self.top_scores = top
        state = self.store.connection.execute(
            "SELECT data FROM " + self.store.table('checkpoint_analysis') + " WHERE scope=? AND scene=? AND model=?",
            (self.store.scope, self.scene, self.model)).fetchone()
        self.analysis = decode(state[0]) if state else None
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
        if self.top_scores is None or self.processed == self.saved_processed:
            return
        db = self.store.connection
        with db:
            db.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(scope,scene,model) DO UPDATE SET signature=excluded.signature,processed=excluded.processed,"
                "total=excluded.total,top_rows=excluded.top_rows,top_scores=excluded.top_scores",
                (self.store.scope, self.scene, self.model, self.signature, self.processed, self.total,
                 len(self.top_scores), zlib.compress(self.top_scores.astype("<f4", copy=False).tobytes())))
            if self.pending:
                db.execute("INSERT INTO checkpoint_frames VALUES (?,?,?,?,?)",
                           (self.store.scope, self.scene, self.model, self.processed, encode(self.pending)))
            if self.analysis is not None:
                db.execute("INSERT INTO checkpoint_analysis VALUES (?,?,?,?) ON CONFLICT(scope,scene,model) "
                           "DO UPDATE SET data=excluded.data", (self.store.scope, self.scene, self.model, encode(self.analysis)))
        self.saved_processed = self.processed
        self.pending, self.batches = [], 0


def prepared_batches(app, tagger, stash, paths, cues, prefetch=False, workers=1):
    """Current batch plus N queued batches; one producer owns media/PIL."""
    if not isinstance(prefetch, int) or not 0 <= prefetch <= 16:
        raise ValueError('prefetch must be an integer in 0..16.')
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('prepare workers must be an integer in 1..16.')
    iterator = app.iter_sprite_frames(stash, paths, cues)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prepare-frames") if prefetch else None
    transforms = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='transform-frames') if workers > 1 else None
    def transform(image):
        try:
            return tagger.prepare(image)
        finally:
            if hasattr(image, 'close'):
                image.close()
    def prepare_next():
        prepared, batch_cues = [], []
        for _ in range(tagger.batch_size):
            try:
                cue, image = next(iterator)
            except StopIteration:
                break
            if transforms:
                # The source iterator closes its frame on the next step. Each
                # worker owns a copy, and closes it even if preparation fails.
                owned = image.copy() if hasattr(image, 'copy') else image
                prepared.append(transforms.submit(transform, owned))
            else:
                prepared.append(tagger.prepare(image))
            batch_cues.append(cue)
        if transforms:
            prepared = [future.result() for future in prepared]
        return prepared, batch_cues
    try:
        pending = deque(executor.submit(prepare_next) for _ in range(prefetch)) if executor else deque()
        while True:
            prepared, batch_cues = pending.popleft().result() if executor else prepare_next()
            if not prepared:
                break
            if executor:
                pending.append(executor.submit(prepare_next))
            yield prepared, batch_cues
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
        if transforms:
            transforms.shutdown(wait=True)
        iterator.close()
