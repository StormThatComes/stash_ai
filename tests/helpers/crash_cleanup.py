"""Durable fake remote mutation followed by immediate process termination."""
import json
import os
from pathlib import Path
import sys

from pytest import MonkeyPatch
from tests import conftest
from tests.helpers.cleanup import options, install
from tests.helpers.cleanup_sql import SQLStash
from tests.helpers.runtime import runtime as app


def main():
    directory, stage = Path(sys.argv[1]), sys.argv[2]
    patch = MonkeyPatch()
    conftest.reset_progress.__wrapped__(patch, directory)
    stash = SQLStash()
    from stash_ai.evidence.labels import AI
    blocked = stash.tag(AI.prefix + 'blocked')
    stash.scene['tags'] = [blocked]
    stash.scene['scene_markers'] = [dict(id='50', title='Manual', seconds=0, end_seconds=2,
                                       primary_tag=blocked, tags=[])]
    install(patch, stash)
    def persist():
        (directory / 'remote.json').write_text(json.dumps(dict(tags=stash.tags, scene=stash.scene)))
    persist()
    original = stash.gql
    def gql(query, variables=None):
        value = original(query, variables)
        if 'CleanupDeleteMarkers' in query:
            persist()
            if stage == 'mutation':
                os._exit(73)
        return value
    patch.setattr(stash, 'gql', gql)
    from stash_ai.cleanup import sql
    original_finish = sql.finish
    def finish(*args):
        original_finish(*args)
        persist()
        if stage == 'finish':
            os._exit(73)
    patch.setattr(sql, 'finish', finish)
    app.main([*options(directory, ['blocked']), '--cleanup-strategy', 'sql'])
    raise AssertionError('Crash point was not reached')


if __name__ == '__main__':
    main()
