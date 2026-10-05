"""Canonical schema facts shared by storage, previews and diagnostics."""
from dataclasses import dataclass

APPLICATION_ID = 0x53544D47
LATEST_SCHEMA_VERSION = 17
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
    TableSpec('inference_archive', ('data',), model=True, kind=True),
    TableSpec('results', ('data',), True, True, base=True, invalidates_merge=True),
    TableSpec('result_thresholds', model=True, kind=True, invalidates_merge=True),
    TableSpec('result_metadata', model=True, kind=True),
    TableSpec('snapshots', ('published', 'pending'), kind=True, base=True, invalidates_merge=True),
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
    from stash_ai.cleanup.state import initialize as initialize_cleanup, load as load_cleanup
    from stash_ai.storage.merge_checks import initialize as initialize_checks
    from stash_ai.storage.recovery import initialize as initialize_recovery
    from stash_ai.storage.preview import initialize as initialize_preview
    from stash_ai.storage.archive import initialize as initialize_archive
    initialize_archive(store)
    initialize_progress(store)
    initialize_cleanup(store)
    if store.preview_source:
        store.connection.execute(
            'INSERT INTO cleanup_state SELECT * FROM cleanup_source.cleanup_state WHERE scope=?',
            (store.scope,))
        store.connection.commit()
        load_cleanup(store)
    initialize_checks(store)
    initialize_recovery(store)
    initialize_preview(store)



def load_features(store):
    """Read current state without issuing schema changes on an ordinary open."""
    from stash_ai.cleanup.state import load as load_cleanup
    from stash_ai.storage.recovery import load as load_recovery
    load_cleanup(store)
    load_recovery(store)


COLUMNS = {
    'inference_archive': 'scope scene model kind signature data',
    'results': 'scope scene model kind data',
    'result_thresholds': 'scope scene model kind thresholds',
    'result_metadata': 'scope scene model kind format data',
    'snapshots': 'scope scene kind published pending',
    'result_index': 'scope scene model kind inference populated signature',
    'refresh_requests': 'scope scene model kind',
    'checkpoints': 'scope scene model signature processed total top_rows top_scores',
    'checkpoint_frames': 'scope scene model processed data',
    'checkpoint_analysis': 'scope scene model data',
    'cleanup_state': 'scope generation data pending',
    'merge_checks': 'scope scene policy fingerprint',
    'recovery_publications': 'scope scene kind operation data',
    'recovery_cleanups': 'scope generation kinds',
}
TRIGGERS = frozenset(
    ['results_delete_threshold']
    + [f'results_metadata_{event}' for event in ('insert', 'update', 'delete')]
    + [f'perf_{table}_{event}' for table in (*MERGE_INVALIDATING_TABLES, 'cleanup')
       for event in ('insert', 'update', 'delete')])


def shape_issues(db):
    """Current SQL requires these column orders, indexes and invalidators."""
    from collections import Counter
    objects = db.execute("SELECT type,name FROM sqlite_master").fetchall()
    tables = {name for kind, name in objects if kind == 'table'}
    issues = Counter(missing_state_tables=len(STATE_TABLES - tables))
    for name in STATE_TABLES & tables:
        columns = [row[1] for row in db.execute(f'PRAGMA table_info({name})')]
        issues['invalid_table_columns'] += int(columns != COLUMNS[name].split())
    issues['missing_state_triggers'] = len(TRIGGERS - {name for kind, name in objects if kind == 'trigger'})
    issues['missing_state_indexes'] = int(('index', 'snapshots_pending_scope') not in objects)
    return +issues
