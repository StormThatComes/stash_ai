"""Current source ownership for a frozen late-stage query proof.

Selected observations shadow durable evidence even when their membership was
removed. Invalidated references stay in the frozen ranking as rivals, but may
not authorize another positive assignment after a query-cohort repair.
"""
from .database import encoded


class FrozenOwnership:
    def __init__(self, selected, durable, aliases):
        self.selected, self.durable, self.aliases = selected, durable, aliases
        self.owners = {}

    def allows(self, owner, identifiers):
        identifiers = tuple(dict.fromkeys(identifiers))
        if not identifiers:
            return False
        missing = set(identifiers).difference(self.owners)
        for connection in (self.selected, self.durable):
            if not missing:
                break
            rows = connection.execute('SELECT o.id,m.person FROM observations o '
                'LEFT JOIN members m ON m.observation=o.id WHERE o.id IN '
                '(SELECT value FROM json_each(?))',(encoded(sorted(missing)),))
            for row in rows:
                self.owners[row['id']] = self.aliases.get(row['person'],row['person'])
                missing.remove(row['id'])
        for identifier in missing:
            self.owners[identifier] = None
        return all(self.owners[i] == owner for i in identifiers)

    def proof_allows(self, owner, proof, groups):
        if proof['mode'] == 'view':
            return bool(proof['view_evidence']) and all(
                self.allows(owner,v['support_ids']) for v in proof['view_evidence'])
        references = tuple(g for g in groups if g.identifier == owner)
        return bool(references) and all(self.allows(owner,g.sample_ids) for g in references)
