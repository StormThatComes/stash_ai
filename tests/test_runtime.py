"""Runtime ownership and explicit service injection regression checks."""
from stash_ai import cli
from stash_ai.commands import dispatch
from stash_ai.runtime.context import RuntimeContext
from stash_ai.runtime.settings import RuntimeSettings
import pytest


def test_default_cli_invocations_own_independent_runtime(monkeypatch):
    runtimes = []

    def database_command(runtime, args):
        assert runtime.DRY_RUN is False
        runtime.DRY_RUN = True
        runtimes.append(runtime)
        return 0

    monkeypatch.setattr(dispatch, "run_database_command", database_command)
    assert cli.main([]) == cli.main([]) == 0
    assert runtimes[0] is not runtimes[1]
    assert runtimes[0].services is not runtimes[1].services


def test_settings_and_services_are_owned_by_the_injected_runtime(monkeypatch):
    first = RuntimeContext(STASH_API_KEY="", STASH_URL="http://first:9999")
    second = RuntimeContext(STASH_API_KEY="", STASH_URL="http://second:9999")
    first.JOYTAG_CLIP_MEAN[0] = 0
    assert second.JOYTAG_CLIP_MEAN[0] != 0
    first.services.create_stash_client = lambda dry_run: (first.STASH_URL, dry_run)
    assert first.services.create_stash_client(True) == ("http://first:9999", True)
    assert second.services.create_stash_client is not first.services.create_stash_client
    monkeypatch.setattr(dispatch, "run_database_command", lambda runtime, args: 7 if runtime is first else 9)
    assert cli.main([], runtime=first) == 7


def test_api_key_has_no_historical_default_and_is_read_for_each_runtime(monkeypatch):
    monkeypatch.delenv('STASH_API_KEY', raising=False)
    assert RuntimeSettings().STASH_API_KEY == ''
    monkeypatch.setenv('STASH_API_KEY', 'configured-test-key')
    assert RuntimeSettings().STASH_API_KEY == 'configured-test-key'
    monkeypatch.setenv('STASH_API_KEY', '')
    assert RuntimeSettings().STASH_API_KEY == ''


@pytest.mark.parametrize('name,value', [
    ('TOP_K_FRAMES', True), ('MARKER_MIN_HITS', False), ('SCENE_PAGE_SIZE', True),
    ('MAX_FRAMES_PER_SCENE', True), ('MARKER_MAX_TAGS_PER_FRAME', False),
    ('CHECKPOINT_BATCHES', True), ('PREFETCH_BATCHES', True), ('PREPARE_WORKERS', False),
    ('RESULT_PREFETCH_BATCHES', True), ('RESULT_PREFETCH_BATCHES', 0), ('RESULT_PREFETCH_BATCHES', 129),
    ('HTTP_RETRIES', True), ('HTTP_RETRIES', -1), ('HTTP_RETRIES', 11),
    ('REQUEST_TIMEOUT', True), ('REQUEST_TIMEOUT', 0), ('REQUEST_TIMEOUT', float('inf')),
    ('REQUEST_TIMEOUT', (1,)), ('REQUEST_TIMEOUT', (1, 2, 3)),
    ('REQUEST_TIMEOUT', (1, float('nan'))), ('REQUEST_TIMEOUT', (True, 1)),
])
def test_runtime_rejects_ambiguous_counts_and_invalid_http_budgets(name, value):
    from stash_ai.runtime.settings import validate_config
    runtime = RuntimeSettings()
    setattr(runtime, name, value)
    with pytest.raises(ValueError, match=name):
        validate_config(runtime)


@pytest.mark.parametrize('timeout', [.1, (1, .1)])
@pytest.mark.parametrize('retries', [0, 10])
def test_runtime_accepts_finite_positive_timeout_and_retry_boundaries(timeout, retries):
    from stash_ai.runtime.settings import validate_config
    validate_config(RuntimeSettings(REQUEST_TIMEOUT=timeout, HTTP_RETRIES=retries))


@pytest.mark.parametrize('input_prefetch', [0, 128])
@pytest.mark.parametrize('result_prefetch', [1, 128])
def test_runtime_accepts_independent_prefetch_boundaries(input_prefetch, result_prefetch):
    from stash_ai.runtime.settings import validate_config
    validate_config(RuntimeSettings(PREFETCH_BATCHES=input_prefetch,
                                    RESULT_PREFETCH_BATCHES=result_prefetch))
