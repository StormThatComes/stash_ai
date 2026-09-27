"""Stash AI application modules; launch the root stash_ai.py script."""
from pathlib import Path

from update_stash_ai import application_guard


# Protect package imports as well as CLI execution from concurrent updates.
_installation_lock = application_guard(Path(__file__).resolve().parent.parent)
