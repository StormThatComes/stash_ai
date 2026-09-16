#!/usr/bin/env python
"""Compatibility launcher. Configuration and implementation now live in stash_ai.py."""
import sys
import stash_ai

if __name__ == "__main__":
    try:
        raise SystemExit(stash_ai.main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {stash_ai.safe_error(exc)}", file=sys.stderr)
        raise SystemExit(1)
else:
    sys.modules[__name__] = stash_ai
