"""Progressively wider joint restart probes on retained parameter domains."""
from .merge_report_policy import parameter
from .merge_solver_domains import registry
from .merge_solver_search import interleave


def joint_probes(config, changes, options):
    domains = registry(config)
    choices = {}
    for key, values in sorted(options.items()):
        if key not in domains:
            continue
        current = changes.get(key, parameter(config.data, key))
        values = sorted({domains[key].project(v) for v in values if type(v) in (int, float, bool)})
        if values:
            choices[key] = list(dict.fromkeys(values[int(f*(len(values)-1))] for f in (0., .5, 1., .25, .75)))
            if current in choices[key]:
                choices[key].remove(current)
    choices = {k: v for k, v in choices.items() if v}
    keys = list(choices)
    if not keys:
        return
    # Alternate narrow routes and broad cross-family combinations. Include
    # local/global and View parameters whenever retained evidence measured them.
    groups = [[k for k in keys if k.startswith(prefix)] for prefix in
              ('recognition.', 'clustering.', 'sapiensid.matching.', 'sapiensid.identities.')]
    groups += [[k for k in keys if 'view' in k], keys]
    def probes(width):
        for turn in range(5):
            for group in groups:
                if not group:
                    continue
                offset = turn % len(group)
                selected = (group[offset:]+group[:offset])[:width]
                yield dict(changes, **{k: choices[k][turn % len(choices[k])] for k in selected})
    yield from interleave(probes(width) for width in dict.fromkeys((2, 4, len(keys))))
