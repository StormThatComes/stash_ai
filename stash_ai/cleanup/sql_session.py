"""Bounded SQL/API session used by cleanup discovery and mutations.

Only the exact SELECT templates in SQL are accepted by the dry-run transport guard.
Mutation helpers still go through Stash GraphQL so hooks and cleanup semantics remain intact.
"""
from __future__ import annotations

from time import monotonic
from stash_ai.runtime.console import activity, status


READ_OPERATION = '''mutation StashAICleanupReadSQL($sql: String!, $args: [Any]) {
    querySQL(sql: $sql, args: $args) { columns rows }
}'''

# tag_id indexes in Stash also contain rowid. For relation tables, paginate by
# rowid, NOT scene_marker_id/scene_id: the latter can sort the full hit set again
# on every page. No COUNT, OFFSET, recursive hierarchy or marker relation resolvers.
OWNER = """AND (? = '' OR (substr(m.title, 1, length(?)) = ? AND EXISTS (
    SELECT 1 FROM scene_markers_tags own CROSS JOIN tags ot ON ot.id = own.tag_id
    WHERE own.scene_marker_id = m.id AND ot.name = ? COLLATE BINARY)))"""
PRIMARY = """FROM scene_markers m JOIN tags t ON t.id = m.primary_tag_id
    WHERE m.primary_tag_id = ? AND t.name = ? COLLATE BINARY """ + OWNER
SQL = {
    'probe': '''SELECT m.id, m.primary_tag_id, m.scene_id, m.title, t.name,
        mt.rowid, mt.tag_id, mt.scene_marker_id, st.rowid, st.tag_id, st.scene_id
        FROM scene_markers m, tags t, scene_markers_tags mt, scenes_tags st WHERE 0''',
    'json': 'SELECT value FROM json_each(?)',
    'bounds': '''SELECT (SELECT coalesce(max(id), 0) FROM scene_markers),
        (SELECT coalesce(max(rowid), 0) FROM scene_markers_tags),
        (SELECT coalesce(max(rowid), 0) FROM scenes_tags)''',
    'primary': 'SELECT m.id ' + PRIMARY + ' AND m.id > ? AND m.id <= ? ORDER BY m.id LIMIT ?',
    'primary_check': 'SELECT m.id ' + PRIMARY + ' AND m.id IN (SELECT value FROM json_each(?))',
    'primary_scenes': '''SELECT m.id, m.scene_id FROM scene_markers m
        WHERE m.primary_tag_id = ? AND m.id > ? AND m.id <= ? ORDER BY m.id LIMIT ?''',
    'extra_scenes': '''SELECT mt.rowid, m.scene_id FROM scene_markers_tags mt
        JOIN scene_markers m ON m.id = mt.scene_marker_id
        WHERE mt.tag_id = ? AND mt.rowid > ? AND mt.rowid <= ? ORDER BY mt.rowid LIMIT ?''',
    'extras': '''SELECT mt.rowid, mt.scene_marker_id FROM scene_markers_tags mt
        JOIN tags t ON t.id = mt.tag_id WHERE mt.tag_id = ? AND t.name = ? COLLATE BINARY
        AND mt.rowid > ? AND mt.rowid <= ? ORDER BY mt.rowid LIMIT ?''',
    'scenes': '''SELECT st.rowid, st.scene_id FROM scenes_tags st
        JOIN tags t ON t.id = st.tag_id WHERE st.tag_id = ? AND t.name = ? COLLATE BINARY
        AND st.rowid > ? AND st.rowid <= ? ORDER BY st.rowid LIMIT ?''',
    'extra_check': '''SELECT mt.scene_marker_id FROM scene_markers_tags mt
        JOIN tags t ON t.id = mt.tag_id WHERE mt.tag_id = ? AND t.name = ? COLLATE BINARY
        AND mt.scene_marker_id IN (SELECT value FROM json_each(?))''',
    'scene_check': '''SELECT st.scene_id FROM scenes_tags st
        JOIN tags t ON t.id = st.tag_id WHERE st.tag_id = ? AND t.name = ? COLLATE BINARY
        AND st.scene_id IN (SELECT value FROM json_each(?))''',
    'tag_check': 'SELECT id, name FROM tags WHERE id IN (SELECT value FROM json_each(?))',
    'marker_metadata': '''SELECT m.id, t.name FROM scene_markers m JOIN tags t ON t.id=m.primary_tag_id
        WHERE m.id IN (SELECT value FROM json_each(?))''',
    'group_usage': '''SELECT mt.tag_id, count(DISTINCT mt.scene_marker_id)
        FROM scene_markers_tags mt
        WHERE mt.scene_marker_id IN (SELECT value FROM json_each(?))
        AND mt.tag_id IN (SELECT value FROM json_each(?)) GROUP BY mt.tag_id''',
    'unused_candidates': '''SELECT t.id FROM tags t WHERE t.id IN (SELECT value FROM json_each(?))
        AND NOT EXISTS (SELECT 1 FROM scene_markers m WHERE m.primary_tag_id=t.id)
        AND NOT EXISTS (SELECT 1 FROM scene_markers_tags mt WHERE mt.tag_id=t.id)
        AND NOT EXISTS (SELECT 1 FROM scenes_tags st WHERE st.tag_id=t.id)''',
}


def is_read_only_cleanup(query, variables):
    """Only the exact SELECT templates above may pass the dry-run write guard."""
    return (query == READ_OPERATION and isinstance(variables, dict)
            and variables.get('sql') in SQL.values())


class CleanupSqlSession:
    """Stash SQL reads, API writes and keyset paging with shared metrics."""

    def __init__(self, stash, stats, *, page_size: int):
        self.stash = stash
        self.stats = stats
        self.page_size = page_size
        self.bounds = None

    def read(self, key, params=()):
        started = monotonic()
        report = getattr(self.stash, 'report', None)
        with activity(report, 'cleanup SELECT: ' + key):
            result = self.stash.gql(READ_OPERATION, {'sql': SQL[key], 'args': list(params)})['querySQL']
        self.stats['cleanup_sql_reads'] += 1
        self.stats['cleanup_sql_read_ms'] += round((monotonic() - started) * 1000)
        rows = result.get('rows')
        if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
            raise ValueError('Invalid SQL cleanup response; no write attempted for this batch.')
        self.stats['cleanup_rows_read'] += len(rows)
        status(report, completed=1, sql_reads=self.stats['cleanup_sql_reads'], sql_rows=self.stats['cleanup_rows_read'])
        return rows

    def probe(self):
        self.read('probe')
        if self.read('json', ('[1]',)) != [[1]]:
            raise RuntimeError('Stash SQL JSON support is incompatible with cleanup.')
        self.bounds = self.read('bounds')[0]

    def write(self, query, variables):
        started = monotonic()
        self.stats['cleanup_api_writes'] += 1
        try:
            return self.stash.gql(query, variables)
        finally:
            self.stats['cleanup_api_write_ms'] += round((monotonic() - started) * 1000)

    def pages(self, key, params, bound):
        cursor = 0
        while True:
            rows = self.read(key, (*params, cursor, bound, self.page_size))
            if not rows:
                return
            previous = cursor
            for row in rows:
                identifier = int(row[0])
                if identifier <= previous or identifier > bound:
                    raise ValueError('Stash returned an invalid cleanup cursor; stopped before mutation.')
                previous = identifier
            # Capture the cursor BEFORE yielding: the caller may delete these rows.
            cursor = previous
            self.stats['cleanup_pages'] += 1
            status(getattr(self.stash, 'report', None), relation=key, pages=self.stats['cleanup_pages'])
            yield rows
            if len(rows) < self.page_size:
                return
