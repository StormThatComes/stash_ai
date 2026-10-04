"""Complete CLI arguments for focused runtime tests."""
from tests.helpers.runtime import runtime as app
from stash_ai.commands.arguments import parse_args
from stash_ai.runtime.report import RunReport


def runtime_args(*, report=None, **overrides):
    args = parse_args(app, [])
    args.report = report if report is not None else RunReport(write=lambda _: None)
    vars(args).update(overrides)
    return args

