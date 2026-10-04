"""Explicit runtime shared by existing reset-profile fixtures."""
from stash_ai.runtime.context import RuntimeContext


class TestRuntime(RuntimeContext):
    __test__ = False

    def main(self, argv=None):
        from stash_ai.cli import main
        return main(argv, runtime=self)

    def build_markers(self, *args, **kwargs):
        from stash_ai.evidence.marker_building import build_markers
        return build_markers(self, *args, **kwargs)

    def outputs_already_present(self, *args, **kwargs):
        from stash_ai.inference.results import outputs_already_present
        return outputs_already_present(self, *args, **kwargs)


runtime = TestRuntime(STASH_API_KEY="")
