"""Historical fingerprint adapter used only to finish an interrupted cleanup."""
from stash_ai.signatures import digest


def historical(tags, cores, kinds):
    values = {'tags': tags, 'cores': cores, 'marker_blocklist': []}
    if kinds != ['scene', 'markers']:
        values['output_kinds'] = kinds
    return digest(values)
