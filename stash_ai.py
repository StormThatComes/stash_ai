#!/usr/bin/env python
"""Run Stash AI from this directory, with the installation lock held."""

if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["recognition"]:
        sys.dont_write_bytecode = True
    if not globals().get("_STASH_INSTALL_LOCKED"):
        from update_stash_ai import launch_application
        raise SystemExit(launch_application(__file__))

    from stash_ai.cli import run
    raise SystemExit(run())
