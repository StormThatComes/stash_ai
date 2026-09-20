#!/usr/bin/env python
"""Compatibility launcher. Configuration and implementation now live in stash_ai.py."""
import sys

if __name__ == "__main__":
    from update_stash_ai import launch_application
    raise SystemExit(launch_application(__file__))
else:
    import stash_ai
    sys.modules[__name__] = stash_ai
