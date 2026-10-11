"""Revalidate bounded cleanup batches and perform confirmed native API mutations."""
import json
from itertools import batched
from stash_ai.runtime.console import status
from stash_ai.runtime.mutations import write as confirmed_write


class SQLMutationExecutor:
    def __init__(self, session, catalog, preview, *, write_size):
        self.session, self.catalog, self.preview = session, catalog, preview
        self.stash, self.stats = session.stash, session.stats
        self.write_size, self.progress = write_size, None

    def tick(self):
        status(getattr(self.stash, 'report', None),
               cleanup_mode='planned' if self.stash.dry_run else 'confirmed',
               cleanup_changes=f"markers={self.stats['markers_deleted']} marker_tags={self.stats['marker_tag_removals']} scene_tags={self.stats['scene_tag_removals']} definitions={self.stats['blacklisted_tags_deleted']}")
        if self.progress is not None:
            self.progress.set_postfix(deleted=self.stats['markers_deleted'],
                removed=self.stats['marker_tag_removals'] + self.stats['scene_tag_removals'], refresh=False)
            self.progress.update(0)

    def _write(self, query, variables, field, expected, message, *, name, rows):
        def perform():
            result = self.session.write(query, variables)[field]
            valid = result is True if expected is True else (
                result is not None and {str(row['id']) for row in result} == expected)
            if not valid:
                raise RuntimeError(message)
            return result
        return confirmed_write(getattr(self.stash, 'report', None), name, perform, rows=rows)

    def delete_primary(self, identifier, name, source=None):
        from stash_ai.publication.groups import PREFIX
        group_ids = [int(i) for i, n in self.catalog.records.items() if n.startswith(PREFIX)]
        prefix, owner = (source.title_prefix, source.owner_tag) if source else ('', '')
        params = (int(identifier), name, prefix, prefix, prefix, owner)
        for rows in self.session.pages('primary', params, self.session.bounds[0]):
            for group in batched((int(row[0]) for row in rows), self.write_size):
                current = group if self.stash.dry_run else [int(row[0]) for row in
                    self.session.read('primary_check', (*params, json.dumps(group)))]
                if not set(current) <= set(group):
                    raise ValueError('Unexpected cleanup marker IDs; stopped before mutation.')
                self.stats['cleanup_markers_changed_or_gone'] += len(group) - len(current)
                if current and not self.stash.dry_run:
                    ids = list(map(str, current))
                    self._write('''mutation StashAICleanupDeleteMarkers($ids: [ID!]!) {
                        sceneMarkersDestroy(ids: $ids)
                    }''', {'ids': ids}, 'sceneMarkersDestroy', True,
                        'Marker deletion was not confirmed; rerun the same cleanup.',
                        name='marker_delete', rows=len(current))
                if self.stash.dry_run and current and group_ids:
                    for tag_id, count in self.session.read('group_usage', (json.dumps(current), json.dumps(group_ids))):
                        self.preview.group_usage_delta[str(tag_id)] -= int(count)
                self.stats['markers_deleted'] += len(current)
                self.tick()

    def remove_associations(self, identifier, name, *, markers):
        key, check, bound = ('extras', 'extra_check', self.session.bounds[1]) if markers else (
            'scenes', 'scene_check', self.session.bounds[2])
        for rows in self.session.pages(key, (int(identifier), name), bound):
            for group in batched(dict.fromkeys(int(row[1]) for row in rows), self.write_size):
                current = group if self.stash.dry_run else [int(row[0]) for row in
                    self.session.read(check, (int(identifier), name, json.dumps(group)))]
                current = list(dict.fromkeys(current))
                if markers and self.stash.dry_run:
                    current = self.preview.survivors(self.session, self.catalog, current)
                if not set(current) <= set(group):
                    raise ValueError('Unexpected cleanup association IDs; stopped before mutation.')
                if current and not self.stash.dry_run:
                    field, input_type = ('bulkSceneMarkerUpdate', 'BulkSceneMarkerUpdateInput') if markers else (
                        'bulkSceneUpdate', 'BulkSceneUpdateInput')
                    self._write('mutation StashAICleanupRemoveTags($input: ' + input_type +
                        '!) { ' + field + '(input: $input) { id } }', {'input': {
                            'ids': list(map(str, current)), 'tag_ids': {'ids': [str(identifier)], 'mode': 'REMOVE'}}},
                        field, set(map(str, current)), 'Tag detachment was not confirmed; rerun the same cleanup.',
                        name='marker_tags' if markers else 'scene_tag_removals', rows=len(current))
                # These count object/tag operations; one object can lose several tags.
                self.stats['marker_tag_removals' if markers else 'scene_tag_removals'] += len(current)
                self.tick()

    def delete_tag(self, identifier, name):
        if not self.stash.dry_run and self.session.read('tag_check', (json.dumps([int(identifier)]),)) != [[int(identifier), name]]:
            self.stats['cleanup_tags_changed_or_gone'] += 1
            return
        # One tag per mutation: a popular tag can have millions of associations.
        if not self.stash.dry_run:
            self._write('''mutation StashAICleanupDeleteTags($ids: [ID!]!) {
                tagsDestroy(ids: $ids)
            }''', {'ids': [str(identifier)]}, 'tagsDestroy', True,
                'Tag deletion was not confirmed; rerun the same cleanup.', name='tag_delete', rows=1)
        self.stats['blacklisted_tags_deleted'] += 1
        self.tick()

