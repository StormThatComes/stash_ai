"""One projection of recorded production decisions for offline reports."""
# Categories describe the recorded assignment path, not whether the face was
# actually used as a reference at the time. That historical sample set is absent.
REASONS = {
    'historical_potential_identity': ('anchor-deferred', 'A whole local cohort is close to an existing frozen appearance; publication of a new identity was deferred.'),
    'shared_view_source_invalidated': ('withheld', 'The final direct auxiliary assignment revoked the owner of the selected source view.'),
    'selected_cohort_identity': ('anchor-view', 'A supported majority cohort matched the frozen selected bank; unmatched outliers remain withheld.'),
    'selected_view_identity': ('anchor-view', 'A complete local core matched a frozen confirmed view without training another view.'),
    'global_reference_attachment': ('anchor-historical-view', 'A historical fragment matched a frozen source-confirmed view and no longer trains strict references.'),
    'pooled_view_identity': ('anchor-historical-view', 'Independent unresolved anchors jointly matched one frozen historical view bank; recovered evidence remains attachment-only.'),
    'provisional_pool_source_invalidated': ('withheld', 'The final auxiliary decision revoked the frozen pool source owner; its claims remain withheld.'),
    'reference_bank_unavailable': ('withheld', 'A superseded identity reference is temporarily unavailable; a possible duplicate birth was deferred.'),
    'historical_cohort_identity': ('anchor-historical-view', 'An independently supported cohort matched the frozen historical bank; unmatched outliers remain withheld.'),
    'late_strict_identity': ('anchor-recovered', 'Matched the final frozen strict catalogue after all selected scenes were replayed.'),
    'shared_view_identity': ('anchor-view', 'A new local hypothesis matched an independently corroborated earlier view; it cannot train strict references.'),
    'historical_ambiguous_identity': ('anchor-deferred', 'Local evidence is coherent, but global identity rivals prevent publishing a new person.'),
    'candidate_group': ('anchor', 'Accepted with a candidate group after identity support resolution; '
                        'the record does not distinguish the seed from later group members.'),
    'known_identity': ('anchor', 'Matched an established identity in the strict matching or merged-observation recheck pass.'),
    'same_scene_identity': ('anchor-same-scene', 'Attached using the supported strict core in the same scene and local thresholds.'),
    'anchor_view_identity': ('anchor-view', 'Attached using a supported anchor-view prototype.'),
    'cross_scene_identity': ('anchor-cross-scene', 'Confirmed against an established identity using frozen cross-scene evidence.'),
    'cross_scene_provisional': ('anchor-cross-scene', 'Confirmed as a provisional identity using independent support across scenes.'),
    'auxiliary_identity': ('auxiliary', 'Attached a pose-limited observation to an established anchor identity after auxiliary support resolution.'),
    'historical_identity': ('anchor-historical', 'Matched a frozen compatible strict identity outside the selected scene scope.'),
    'historical_view_identity': ('anchor-historical-view', 'Matched an independently confirmed frozen view outside the selected scope.'),
    'global_view_identity': ('anchor-global-view', 'Attached using an independently confirmed frozen strict view from the selected library.'),
    'view_identity': ('auxiliary-view', 'Attached a pose-limited observation using a confirmed view prototype and independent support.'),
}

# Retain numeric evidence and observation pointers, excluding arbitrary future
# payloads, URLs, image data, vectors, installation tokens and run configuration.
DECISION_FIELDS = (
    'shared_reference', 'cohort_resolution', 'strict_partition', 'strict_confirmation', 'reference_registry', 'provisional_pool',
    'global_identity_decision', 'decision_conflict',
    'observation', 'reason', 'scores', 'policy', 'representative_limit', 'reference_role', 'reference_selected',
    'resource_limits', 'reference_context', 'identity_basis', 'merged_from',
    'same_scene_merge', 'same_scene_context', 'cross_scene', 'view_prototype',
    'anchor_view', 'anchor_view_probe', 'anchor_view_seeded',
    'auxiliary_view_probe', 'auxiliary_view_seeded', 'merge_withheld_reason', 'global_view', 'global_view_merge', 'global_view_consolidation', 'historical_reference', 'prior_reason',
)


def recorded(payload):
    return {key: payload[key] for key in DECISION_FIELDS if key in payload}


def explanation(payload):
    return REASONS.get(payload.get("reason"), ("unknown",
        "No built-in explanation for this recorded reason; inspect the recorded evidence."))[1]


def summary(db, policy, scans, *, examples=3):
    """Counts and bounded original evidence; never reconstruct a past decision."""
    import json
    from .database import encoded
    from .reference_roles import observation_sql
    condition = 'policy=? AND scan IN (SELECT value FROM json_each(?))'
    args = (policy, encoded(list(scans)))
    groups = []
    for row in db.execute("SELECT status,json_extract(payload,'$.reason') AS reason,COUNT(*) AS total "
                          'FROM candidates WHERE ' + condition + ' GROUP BY status,reason ORDER BY total DESC,status,reason', args):
        evidence = [recorded(json.loads(r[0])) for r in db.execute('SELECT payload FROM candidates WHERE ' + condition +
            " AND status=? AND json_extract(payload,'$.reason') IS ? ORDER BY " + observation_sql('candidates') + ' LIMIT ?',
            args + (row['status'], row['reason'], examples))]
        groups.append(dict(status=row['status'], reason=row['reason'], count=row['total'], examples=evidence))
    return dict(format=1, matching_policy=policy, groups=groups)
