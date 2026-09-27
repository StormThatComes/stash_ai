"""Set-based Stash scene snapshots. SELECT only; native mutations stay unchanged."""
import json
from stash_ai.runtime import measured

OPERATION = '''mutation StashAIMergeReadSQL($sql: String!, $args: [Any]) {
    querySQL(sql: $sql, args: $args) { columns rows }
}'''

TAGS = """(SELECT json_group_array(json_object('id',CAST(t.id AS TEXT),'name',t.name))
    FROM scenes_tags st JOIN tags t ON t.id=st.tag_id WHERE st.scene_id=s.id)"""
EXTRAS = """(SELECT json_group_array(json_object('id',CAST(et.id AS TEXT),'name',et.name))
    FROM scene_markers_tags mt JOIN tags et ON et.id=mt.tag_id WHERE mt.scene_marker_id=m.id)"""
MARKERS = """(SELECT json_group_array(json_object('id',CAST(m.id AS TEXT),
    'title',m.title,'seconds',m.seconds,'end_seconds',{end},
    'primary_tag',json_object('id',CAST(t.id AS TEXT),'name',t.name),'tags',{extras}))
    FROM scene_markers m JOIN tags t ON t.id=m.primary_tag_id WHERE m.scene_id=s.id)"""
SQL = {mode: "SELECT s.id,json_object('id',CAST(s.id AS TEXT),'title',s.title,'tags'," + TAGS +
    ",'scene_markers'," + (MARKERS.format(end='m.end_seconds' if mode == 'ranges' else 'NULL', extras=EXTRAS)
        if mode != 'tags' else "json('[]')") + ") AS snapshot FROM scenes s " +
    "WHERE s.id IN (SELECT value FROM json_each(?)) ORDER BY s.id"
    for mode in ('ranges', 'points', 'tags')}
SQL.update({'page_' + mode: sql.replace(
    'WHERE s.id IN (SELECT value FROM json_each(?)) ORDER BY s.id',
    'WHERE s.id > ? ORDER BY s.id LIMIT ?') for mode, sql in list(SQL.items())})


def is_read_only_merge(query, variables):
    return (query == OPERATION and isinstance(variables, dict)
            and variables.get('sql') in SQL.values())


class SceneReader:
    def __init__(self, stash, markers, report=None):
        self.stash, self.report = stash, report
        self.mode = 'ranges' if markers and stash.has_end_time else 'points' if markers else 'tags'

    def read(self, identifiers):
        wanted = {int(identifier) for identifier in identifiers}
        if any(value <= 0 for value in wanted):
            raise ValueError('Invalid scene ID for merge snapshot')
        return self._read(self.mode, [json.dumps(sorted(wanted))], wanted)

    def page(self, after, size):
        if type(after) is not int or after < 0 or type(size) is not int or not 1 <= size <= 1000:
            raise ValueError('Invalid merge scene page')
        scenes = self._read('page_' + self.mode, [after, size])
        if len(scenes) > size or any(int(i) <= after for i in scenes):
            raise ValueError('Invalid merge scene page cursor')
        return scenes

    @measured('stash_snapshot_decode')
    def _read(self, mode, params, wanted=None):
        result = self.stash.gql(OPERATION, {'sql': SQL[mode], 'args': params})['querySQL']
        rows = result.get('rows')
        if not isinstance(rows, list):
            raise ValueError('Invalid SQL merge snapshot response')
        scenes = {}
        for row in rows:
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError('Invalid SQL merge snapshot row')
            identifier = int(row[0])
            scene = json.loads(row[1])
            if (identifier <= 0 or (wanted is not None and identifier not in wanted) or str(identifier) in scenes or not isinstance(scene, dict)
                    or scene.get('id') != str(identifier) or not isinstance(scene.get('tags'), list)
                    or not isinstance(scene.get('scene_markers'), list)):
                raise ValueError('Unexpected scene in SQL merge snapshot')
            scenes[str(identifier)] = scene
        if self.report is not None:
            self.report.counts['merge_sql_reads'] += 1
        return scenes
