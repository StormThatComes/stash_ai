"""Resolve explicit negative samples against the current retained replay inputs."""


def resolve(db, config, pairs, scope, owners, *, positive=False, physical_pairs=None):
    if not pairs:
        return []
    from .identity_report import _select
    from .models import namespace
    from .offline_evaluation import _selection
    from .body_identity import enabled, available as identities_available
    from .body_observations import available, contract

    constraints, selected, performers = {}, {}, {}
    label = 'Positive' if positive else 'Excluded'
    bodies = available(db)
    body_enabled = enabled(config) and bodies and identities_available(db)
    expected = contract(config)
    physical_pairs = physical_pairs or {}
    manual_faces = {body:face for face,body in physical_pairs.items()}

    def sample(oid):
        rows = [dict(r, evidence_kind='face') for r in db.execute(
            'SELECT o.id,o.scan,o.cue,s.scene,s.scope,s.namespace FROM observations o '
            'JOIN scans s ON s.id=o.scan WHERE o.id=?', (oid,))]
        if bodies:
            rows.extend(dict(r, evidence_kind='person') for r in db.execute(
                'SELECT b.id,b.scan,b.cue,b.face,b.contract,s.scene,s.scope,s.namespace '
                'FROM body_observations b JOIN scans s ON s.id=b.scan WHERE b.id=?', (oid,)))
        if len(rows) != 1:
            raise ValueError(f'{label} observation {oid} is unknown or ambiguous; use a current report-identity manifest.')
        row = rows[0]
        if row['scope'] != scope or row['namespace'] != namespace(config):
            raise ValueError(f'{label} observation {oid} belongs to another scope or recognizer namespace.')
        if row['evidence_kind'] == 'person' and (not body_enabled or row['contract'] != expected):
            raise ValueError(f'{label} observation {oid} requires enabled body identities with a compatible contract.')
        if row['scene'] not in selected:
            _, scans = _selection(db, config, [row['scene']], scope)
            selected[row['scene']] = scans[0]['id']
        if row['scan'] != selected[row['scene']]:
            raise ValueError(f'{label} observation {oid} is not in the current scan; regenerate report-identity.')
        return row

    for performer, oid in pairs:
        if (not isinstance(performer, str) or not performer.isascii() or not performer.isdecimal()
                or int(performer) < 1 or not isinstance(oid, str) or not oid or ':' in oid
                or any(c.isspace() for c in oid)):
            raise ValueError('Observation exclusions require PERFORMER_ID:OBSERVATION_ID.')
        performer = str(int(performer))
        if performer not in performers:
            _, performers[performer] = _select(db, namespace(config), None, performer, scope)
        person = performers[performer]
        if person['id'] not in owners:
            raise ValueError(f'{label} performer {performer} must belong to a merge goal set.')
        row = sample(oid)
        ids = {oid}
        face = row['id'] if row['evidence_kind'] == 'face' else row['face']
        if row['evidence_kind']=='person':
            if oid in manual_faces:
                face = manual_faces[oid]
            elif face in physical_pairs:
                face = None  # The retained body was displaced by a manual correction.
        if face:
            linked = sample(face)
            if linked['evidence_kind'] != 'face' or (linked['scan'], linked['cue']) != (row['scan'], row['cue']):
                raise ValueError(f'{label} observation {oid} has an incompatible linked face.')
            ids.add(face)
            if face in physical_pairs:
                ids.add(physical_pairs[face])
            elif body_enabled:
                # The retained association defines this physical sample. Do not
                # propagate a label to other people in the cue or to other cues.
                ids.update(r[0] for r in db.execute('SELECT id FROM body_observations '
                    'WHERE face=? AND scan=? AND cue=? AND contract=?',
                    (face, row['scan'], row['cue'], expected)) if r[0] not in manual_faces)
        group = owners[person['id']]
        key = group, tuple(sorted(ids))
        item = constraints.setdefault(key, dict(set=group, scene=row['scene'], scan=row['scan'], **({'cue':row['cue']} if positive else {}),
            performer_ids=[], resolved_identity_ids=[], requested_observation_ids=[], observation_ids=sorted(ids)))
        item['performer_ids'] = sorted(set(item['performer_ids']) | {performer})
        item['resolved_identity_ids'] = sorted(set(item['resolved_identity_ids']) | {person['id']})
        item['requested_observation_ids'] = sorted(set(item['requested_observation_ids']) | {oid})
    return [constraints[k] for k in sorted(constraints)]
