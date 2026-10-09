"""Hard exits in the real CLI with the same offline fixtures as pipeline tests."""
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import sys

from pytest import MonkeyPatch
from tests import conftest
from tests.helpers.cleanup import FastStash
from tests.helpers.runtime import runtime as app


def main():
    directory, stage, cadence = sys.argv[1:]
    directory, cadence = Path(directory), int(cadence)
    patch = MonkeyPatch()
    conftest.reset_progress.__wrapped__(patch, directory)
    pipeline = conftest.pipeline.__wrapped__(patch, directory)
    stash = FastStash()
    stash.scenes['2'] = dict(deepcopy(stash.scene), id='2')
    patch.setattr(app.services, 'create_stash_client', lambda _: stash)
    original_settings = app.services.load_model_settings
    patch.setattr(app.services, 'load_model_settings',
                  lambda *a, **k: [replace(value, batch_size=2) for value in original_settings(*a, **k)])
    from stash_ai.inference import checkpoints
    from stash_ai.history.store import HistoryStore
    def exit_at(name):
        if stage == name:
            os._exit(73)
    original_flush = checkpoints.Checkpoint._flush
    def flush(checkpoint):
        original_flush(checkpoint)
        if checkpoint.saved_processed:
            exit_at('flush')
    patch.setattr(checkpoints.Checkpoint, '_flush', flush)
    original_finish = checkpoints.finish
    def finish(*args):
        exit_at('before_finish')
        original_finish(*args)
        exit_at('finish')
    patch.setattr(checkpoints, 'finish', finish)
    original_history = HistoryStore.finish
    def finish_history(*args, **kwargs):
        exit_at('before_history_finish')
        original_history(*args, **kwargs)
        exit_at('history_finish')
    patch.setattr(HistoryStore, 'finish', finish_history)
    app.main(['--models', 'wd', '--no-hierarchy', '--state-db', str(pipeline.path),
              '--merge-workers', '1',
              '--checkpoint-batches', str(cadence),
              '--prefetch-batches', '2', '--result-prefetch-batches', '8'])
    raise AssertionError('Crash point was not reached')


if __name__ == '__main__':
    main()
