"""Subprocess-only abrupt publication exits with durable fake Stash responses."""
import json
import os
from pathlib import Path
import sys

from stash_ai.runtime.settings import RuntimeSettings
from stash_ai.evidence.temporal.policy import TemporalPolicy
from stash_ai.storage.store import StateStore
from stash_ai.evidence.pipeline import evidence
from stash_ai.publication.merged import sync_merged
from stash_ai.stash_api.tags import TagResolver
from stash_ai.evidence.labels import AI
from tests.helpers.stash import Stash
from tests.helpers.evidence import empty_policies


def main():
    state_path, scene_path, settings_path, scope, stage = sys.argv[1:]
    scene_path = Path(scene_path)
    data = json.loads(scene_path.read_text())
    values = json.loads(Path(settings_path).read_text())
    values['TEMPORAL_POLICY'] = TemporalPolicy.from_snapshot(values['TEMPORAL_POLICY'])
    app = RuntimeSettings(**values)
    stash = Stash()
    stash.tags, stash.scene = data['tags'], data['scene']
    def exit_at(name):
        if stage == name:
            os._exit(73)
    def persist(name, method):
        def call(*args, **kwargs):
            value = method(*args, **kwargs)
            scene_path.write_text(json.dumps(dict(tags=stash.tags, scene=stash.scene)))
            exit_at(name)
            return value
        return call
    for name, operation in [('create', 'create_marker'), ('update', 'update_marker'),
                            ('scene', 'update_scene_tags'), ('delete', 'delete_markers')]:
        setattr(stash, operation, persist(name, getattr(stash, operation)))
    with StateStore(state_path, scope) as store:
        store.prepare = persist('prepare', store.prepare)
        commit = store.commit_prepared
        def commit_prepared(*args, **kwargs):
            exit_at('before_commit')
            result = commit(*args, **kwargs)
            exit_at('commit')
            return result
        store.commit_prepared = commit_prepared
        hits, spans = evidence(app, store, '1', empty_policies(), empty_policies())
        sync_merged(app, stash, TagResolver(stash, source=AI), store, stash.find_scene('1'),
                    hits, spans, tag_policies=empty_policies())


if __name__ == '__main__':
    main()
