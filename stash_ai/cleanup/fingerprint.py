"""Identity of a cleanup intent, including its output scope and complete policies."""
from stash_ai.models import SOURCES
from stash_ai.signatures import digest


def fingerprint(tags, cores, kinds):
    return digest({'tags': {s: sorted(tags[s]) for s in SOURCES},
                   'cores': {s: sorted(cores[s]) for s in SOURCES},
                   'output_kinds': kinds})
