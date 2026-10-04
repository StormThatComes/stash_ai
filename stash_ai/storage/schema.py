"""Canonical schema facts shared by storage, previews and diagnostics."""
from dataclasses import dataclass

APPLICATION_ID = 0x53544D47
LATEST_SCHEMA_VERSION = 12
KNOWN_KINDS = frozenset(('scene', 'markers'))
KNOWN_MODELS = frozenset(('joytag', 'wd', 'dtq', 'oppai'))


@dataclass(frozen=True)
class TableSpec:
    name: str
    blobs: tuple[str, ...] = ()
    model: bool = False
    kind: bool = False
    preview: bool = True
    base: bool = False
    invalidates_merge: bool = False


TABLES = (
    TableSpec('results', ('data',), True, True, base=True, invalidates_merge=True),
    TableSpec('result_thresholds', model=True, kind=True, invalidates_merge=True),
    TableSpec('result_metadata', model=True, kind=True),
    TableSpec('snapshots', ('published', 'pending'), kind=True, base=True, invalidates_merge=True),
    TableSpec('legacy_imports', ('data',), kind=True, base=True),
    TableSpec('result_index', model=True, kind=True),
    TableSpec('refresh_requests', model=True, kind=True),
    TableSpec('checkpoints', ('top_scores',), model=True),
    TableSpec('checkpoint_frames', ('data',), model=True),
    TableSpec('checkpoint_analysis', ('data',), model=True),
    TableSpec('cleanup_state', ('data',), preview=False),
    TableSpec('merge_checks', preview=False),
    TableSpec('recovery_publications', ('data',), kind=True),
    TableSpec('recovery_cleanups', preview=False),
)
STATE_TABLES = frozenset(table.name for table in TABLES)
BASE_TABLES = frozenset(table.name for table in TABLES if table.base)
BLOB_COLUMNS = {table.name: table.blobs for table in TABLES if table.blobs}
PREVIEW_TABLES = tuple(table.name for table in TABLES if table.preview)
MERGE_INVALIDATING_TABLES = tuple(table.name for table in TABLES if table.invalidates_merge)


def initialize_features(store):
    """Create current feature tables and load state before installing read views."""
    from stash_ai.storage.progress import initialize_progress
    from stash_ai.cleanup.state import initialize as initialize_cleanup
    from stash_ai.storage.merge_checks import initialize as initialize_checks
    from stash_ai.storage.recovery import initialize as initialize_recovery
    from stash_ai.storage.preview import initialize as initialize_preview
    initialize_progress(store)
    initialize_cleanup(store)
    if store.preview_source:
        store.connection.execute(
            'INSERT INTO cleanup_state SELECT * FROM cleanup_source.cleanup_state WHERE scope=?',
            (store.scope,))
        store.connection.commit()
        initialize_cleanup(store)
    initialize_checks(store)
    initialize_recovery(store)
    initialize_preview(store)
