"""Runtime ownership and explicit service injection regression checks."""
from stash_ai import cli
from stash_ai.commands import dispatch
from stash_ai.runtime.context import RuntimeContext


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
