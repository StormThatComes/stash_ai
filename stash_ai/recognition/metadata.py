"""Automatic metadata extension, independent name and portrait outcomes."""
from collections import Counter

from stash_ai.stash_api.transport import safe_error
from .names import sync_name
from .portraits import fill_portrait


def enrich(database, client, report, config, plan, scope, target, paths, services):
    counts = Counter()
    for name, action in (
        ('names', lambda: sync_name(database, client, report, scope, target, plan)),
        ('portraits', lambda: fill_portrait(database, client, report, config, scope, target, paths,
                                           services.backend, services.decoder)),
    ):
        try:
            counts.update(action())
        except KeyboardInterrupt:
            raise
        except Exception as error:
            counts[name + '_failed'] += 1
            counts['metadata_failed'] += 1
            report.optional('recognition_metadata', report.write,
                            f'Recognition {name} withheld: ' + safe_error(error, client.api_key))
    return counts
