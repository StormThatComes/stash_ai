"""CI-only pytest plugin. Keep whole modules together and use normal pytest discovery."""
from bisect import bisect_right


# First file of shards 2 and 3, based on elapsed time in Actions runs
# 1013, 1024 and 1026. The Linux JUnit artifacts carry timings for rebalancing.
BOUNDARIES = (
    'tests/test_recognition_gpu_memory.py',
    'tests/test_recognition_replay_checkpoints.py',
)


def pytest_addoption(parser):
    parser.addoption('--ci-shard', choices=('1', '2', '3'), help='Linux CI shard to collect')


def shard_number(relative_path):
    return 1 + bisect_right(BOUNDARIES, relative_path)


def pytest_ignore_collect(collection_path, config):
    shard = config.getoption('--ci-shard')
    if shard is not None and collection_path.is_file():
        path = collection_path.relative_to(config.rootpath).as_posix()
        if shard_number(path) != int(shard):
            return True
    return None
