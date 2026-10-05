"""Exit handling for explicitly invoked maintenance tools."""
import sys

from stash_ai.runtime.context import RuntimeContext
from stash_ai.stash_api.transport import safe_error


def run(main):
    runtime = RuntimeContext()
    try:
        return main(runtime=runtime)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {safe_error(exc, api_key=runtime.STASH_API_KEY)}", file=sys.stderr)
        return 1
