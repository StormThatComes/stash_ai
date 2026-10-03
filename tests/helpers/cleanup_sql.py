"""Shared offline helpers; no test collection or fixture registration."""
from copy import deepcopy
import sqlite3

from stash_ai.cleanup.sql_session import READ_OPERATION, SQL
from tests.helpers.cleanup import FastStash


SCHEMA = '''
CREATE TABLE tags(id INTEGER PRIMARY KEY, name TEXT UNIQUE);
CREATE TABLE scene_markers(id INTEGER PRIMARY KEY, title TEXT, primary_tag_id INTEGER, scene_id INTEGER);
CREATE TABLE scene_markers_tags(scene_marker_id INTEGER, tag_id INTEGER);
CREATE TABLE scenes_tags(scene_id INTEGER, tag_id INTEGER);
CREATE INDEX index_scene_markers_on_primary_tag_id ON scene_markers(primary_tag_id);
CREATE INDEX index_scene_markers_on_scene_id ON scene_markers(scene_id);
CREATE INDEX index_scene_markers_tags_on_tag_id ON scene_markers_tags(tag_id);
CREATE INDEX index_scene_markers_tags_on_scene_marker_id ON scene_markers_tags(scene_marker_id);
CREATE INDEX index_scenes_tags_on_tag_id ON scenes_tags(tag_id);
'''


class SQLStash(FastStash):
    def __init__(self, dry=False):
        super().__init__(dry)
        self.sql_calls = []
        self.before_sql = None

    def database(self):
        db = sqlite3.connect(':memory:')
        db.executescript(SCHEMA)
        db.executemany('INSERT INTO tags VALUES (?,?)', [(int(t['id']), t['name']) for t in self.tags.values()])
        for scene in self.scenes.values():
            db.executemany('INSERT INTO scenes_tags VALUES (?,?)', [(int(scene['id']), int(t['id'])) for t in scene['tags']])
            for marker in scene['scene_markers']:
                db.execute('INSERT INTO scene_markers VALUES (?,?,?,?)',
                    (int(marker['id']), marker['title'], int(marker['primary_tag']['id']), int(scene['id'])))
                db.executemany('INSERT INTO scene_markers_tags VALUES (?,?)',
                    [(int(marker['id']), int(t['id'])) for t in marker['tags']])
        return db

    def gql(self, query, variables=None):
        if 'PruneCandidates' in query:
            result = super().gql('query StashAIPruneCatalog($filter: FindFilterType!) { unused }',
                                 {'filter': {'page':1,'per_page':100000}})
            tags = [t for t in result['findTags']['tags'] if t['id'] in variables['ids']]
            for tag in tags:
                tag['image_count'] = sum(any(t['id'] == tag['id'] for t in image['tags']) for image in self.images)
            return {'findTags': {'tags': tags}}
        if query == READ_OPERATION:
            key = next(k for k, v in SQL.items() if v == variables['sql'])
            self.sql_calls.append((key, deepcopy(variables['args'])))
            if self.before_sql:
                self.before_sql(self, key)
            with self.database() as db:
                cursor = db.execute(variables['sql'], variables['args'])
                return {'querySQL': {'columns': [d[0] for d in cursor.description], 'rows': [list(row) for row in cursor]}}
        if 'CleanupRemoveTags' in query:
            assert not self.dry_run
            payload = variables['input']
            assert payload['tag_ids']['mode'] == 'REMOVE'
            ids, tags = set(payload['ids']), set(payload['tag_ids']['ids'])
            field = 'bulkSceneMarkerUpdate' if 'BulkSceneMarkerUpdateInput' in query else 'bulkSceneUpdate'
            objects = [m for s in self.scenes.values() for m in s['scene_markers']] if field == 'bulkSceneMarkerUpdate' else list(self.scenes.values())
            changed = []
            for obj in objects:
                if str(obj['id']) in ids:
                    obj['tags'] = [t for t in obj['tags'] if str(t['id']) not in tags]
                    changed.append({'id': str(obj['id'])})
            self.writes.append(('remove_tags', deepcopy(payload)))
            if self.fail_after == 'CleanupRemoveTags':
                self.fail_after = None
                raise RuntimeError('Connection lost after committed removal')
            return {field: changed}
        return super().gql(query, variables)
