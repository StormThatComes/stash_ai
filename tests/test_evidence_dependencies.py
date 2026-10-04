"""The low-level geometry layer must not load temporal or occurrence algorithms."""
import subprocess
import sys


def test_geometry_import_does_not_load_evidence_algorithms():
    subprocess.run([
        sys.executable, '-c',
        "import sys; from stash_ai.evidence.interval_geometry import union_intervals; "
        "assert 'stash_ai.evidence.occurrences' not in sys.modules; "
        "assert 'stash_ai.evidence.temporal.extras' not in sys.modules; "
        "assert union_intervals([]) == []",
    ], check=True)
