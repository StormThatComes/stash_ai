"""Membership is not authority to train strict identity references.

The latest derived decision describes the current membership. Old databases
with membership-only evidence keep their legacy anchor behavior; an explicit
contextual decision can never silently acquire strict reference authority.
"""

STRICT_REASONS = ('candidate_group', 'known_identity', 'cross_scene_identity',
                  'cross_scene_provisional', 'historical_identity')


def observation_sql(alias):
    return (f"CASE WHEN json_valid({alias}.payload) THEN "
            f"json_extract({alias}.payload,'$.observation') END")


def strict_sql(observation='o', member='m'):
    reasons = ','.join(repr(reason) for reason in STRICT_REASONS)
    return (f"COALESCE((SELECT CASE WHEN c.status='accepted' AND c.person={member}.person "
            f"AND json_extract(c.payload,'$.reason') IN ({reasons}) THEN 1 ELSE 0 END "
            f"FROM candidates c WHERE {observation_sql('c')}={observation}.id "
            "ORDER BY c.rowid DESC LIMIT 1),1)")


def index(db):
    # Disposable malformed candidate JSON must not prevent evidence recovery.
    db.execute('CREATE INDEX IF NOT EXISTS candidates_reference_observation '
               'ON candidates(' + observation_sql('candidates').replace('candidates.', '') + ')')
    db.execute("CREATE INDEX IF NOT EXISTS candidates_unresolved_pool ON candidates("
               "CASE WHEN json_valid(payload) THEN json_extract(payload,'$.reason') END,scan) "
               "WHERE status='withheld'")


def verified_sql(observation='o',member='m'):
    """Only direct strict-matched auxiliary confirmations can seed a view."""
    return (f"COALESCE((SELECT c.status='accepted' AND c.person={member}.person "
            "AND json_extract(c.payload,'$.reason')='auxiliary_identity' "
            f"FROM candidates c WHERE {observation_sql('c')}={observation}.id "
            "ORDER BY c.rowid DESC LIMIT 1),0)")


def attachment_sql(observation='o',member='m'):
    """Only currently accepted non-strict, non-direct-auxiliary memberships."""
    reasons=','.join(repr(r) for r in STRICT_REASONS+('auxiliary_identity',))
    return (f"COALESCE((SELECT c.status='accepted' AND c.person={member}.person "
            f"AND json_extract(c.payload,'$.reason') NOT IN ({reasons}) "
            f"FROM candidates c WHERE {observation_sql('c')}={observation}.id "
            "ORDER BY c.rowid DESC LIMIT 1),0)")
