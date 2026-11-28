"""Bind stable face labels through independent overlap components."""
from collections import defaultdict


def bind_votes(rows, mapping, used, performers):
    """Respect existing bindings, then maximize retained labels and overlap."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching

    edges = defaultdict(dict)
    reverse = defaultdict(set)
    for age, row in enumerate(rows):
        cluster, person, performer = row['cluster'], row['person'], row['performer']
        if row['retired'] or cluster in mapping or person in used or (performer and performer in performers):
            continue
        label = ('performer', performer) if performer else ('person', person)
        options = edges[cluster].setdefault(label, [])
        options.append((row['total'], age, person, performer))
        reverse[label].add(cluster)

    remaining = set(edges)
    for seed in sorted(edges):
        if seed not in remaining:
            continue
        pending = [seed]
        remaining.remove(seed)
        clusters, labels = [], set()
        while pending:
            cluster = pending.pop()
            clusters.append(cluster)
            for label in edges[cluster].keys() - labels:
                labels.add(label)
                neighbours = reverse[label] & remaining
                remaining.difference_update(neighbours)
                pending.extend(neighbours)
        if len(clusters) == 1:
            # A genuine consolidation keeps the oldest admissible origin.
            choices = (item for options in edges[seed].values() for item in options)
            assignments = [(seed, min(choices, key=lambda item: item[1]))]
        else:
            clusters.sort()
            ordered = sorted(labels)
            positions = {label: i for i, label in enumerate(ordered)}
            best = {(i,positions[label]): max(options, key=lambda v: (v[0],-v[1]))
                    for i,cluster in enumerate(clusters) for label,options in edges[cluster].items()}
            cardinality = 1 + sum(item[0] for item in best.values())
            weights = {edge: cardinality + item[0] + 1 for edge,item in best.items()}
            for i in range(len(clusters)):
                weights[i,len(labels)+i] = 1
            row_indices, columns = zip(*weights)
            matrix = csr_matrix((list(weights.values()), (row_indices, columns)),
                                shape=(len(clusters),len(labels)+len(clusters)))
            rows_used, columns_used = min_weight_full_bipartite_matching(matrix, maximize=True)
            assignments = [(clusters[i], best[i,j]) for i,j in zip(rows_used,columns_used) if j<len(labels)]
        for cluster, (_, _, person, performer) in assignments:
            mapping[cluster] = person
            used.add(person)
            if performer:
                performers.add(performer)
