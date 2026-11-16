"""Read-only explanations for selected strict view cohorts and hybrid witnesses."""
from contextlib import closing
from io import BytesIO

from .clustering import vector
from .database import encoded
from .inspection_policy import selection
from .inspection_runtime import check_cancelled, open_readonly
from .person_signal import contract
from .pose_policy import anchor_sql
from .reference_roles import strict_sql
from .shared_person_evidence import expression


def face_rows(reader, scope, owner, *, keys=None, offset=0, limit=80):
    """Revalidate owner, active scan and model for every selected address."""
    current, args = selection(reader.config,'face')
    where = 'm.person=? AND s.scope=? AND s.namespace=? AND '+current
    args = [owner,scope,reader.namespace,*args]
    fields = 'o.id,s.scene,o.cue'
    if keys is not None:
        where += ' AND o.id IN (SELECT value FROM json_each(?))'
        args.append(encoded(list(keys)))
        fields += (',o.scan,o.seconds,o.geometry,o.quality,o.dimensions,o.embedding,o.rejected,'
                   +anchor_sql('o',reader.config.data['clustering'])+' AS quality_ok,'+strict_sql()+' AS strict_ok')
    with closing(open_readonly(reader.config.database)) as db:
        return [dict(r) for r in db.execute('SELECT '+fields+' FROM members m JOIN observations o ON o.id=m.observation '
            'JOIN scans s ON s.id=o.scan WHERE '+where+' ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.geometry,o.embedding,o.id '
            'LIMIT ? OFFSET ?',(*args,limit,offset))]


def formation(reader, scope, owner, keys):
    from .view_prototypes import add_cohort, witness_limit, coverage_threshold
    from .evidence_pixels import alias_scope
    from .inspection_runtime import stamp
    config = reader.config
    revision = stamp(config.database)
    rows = face_rows(reader,scope,owner,keys=keys,limit=len(keys))
    if len(rows) != len(set(keys)):
        raise ValueError('Selected faces or memberships changed. Refresh the comparison.')
    policy = config.data['clustering']
    notes = []
    with closing(open_readonly(config.database)) as db, alias_scope(db,scope,reader.namespace):
        eligible = []
        for row in rows:
            if row['rejected'] or not row['embedding'] or not row['quality_ok'] or not row['strict_ok']:
                reason = row['rejected'] or ('missing embedding' if not row['embedding'] else
                                            'anchor quality gate' if not row['quality_ok'] else 'attachment-only membership')
                notes.append(f"Scene {row['scene']} cue {row['cue']} · {row['id']} · {reason}")
            else:
                eligible.append(row)
        for title, floor in [('View cohesion',policy['view_cohesion_threshold']),('Appearance coverage',coverage_threshold(policy))]:
            slots = []
            for row in eligible:
                check_cancelled()
                add_cohort(slots,row,vector(row),floor,complete=True,maximum=witness_limit(policy))
            notes.append(f'{title} {floor:g} · {len(slots)} coherent groups from {len(eligible)} eligible faces.')
            for slot in slots:
                if slot.needs_final_selection:
                    slot._resolve(slot.maximum)
                votes = len(slot.cues)
                required = policy['view_min_support']
                notes.append(f"{slot.pose} · {votes}/{required} independent votes · " +
                             ('eligible for promotion' if votes >= required else f'{required-votes} more needed'))
                notes.append('Independent witnesses: '+', '.join(r[0] for r in slot.cue_representatives.values()))
            if len(slots)>1:
                notes.append('Selected faces split at the pose or complete-link cohesion gate. Votes belong to each group.')
    if revision != stamp(config.database):
        raise ValueError('Recognition changed during the check. Refresh the comparison.')
    return '\n'.join(notes)


def _signal_pair(db, config, groups, signal, minimum):
    """Locate a reference pair at the reported floor, within both diagnostic budgets."""
    import numpy as np
    if len(groups)!=2 or not all(groups):
        return (None,None), 'No compatible witnesses'
    examples = (groups[0][0],groups[1][0])
    fallback = 'Catalogue examples · limiting witnesses unavailable'
    count = len(groups[0])*len(groups[1])
    budgets = config.data['sapiensid']['execution']
    if minimum is None or count > budgets['max_witness_comparisons'] or (sum(map(len,groups))*17000 > budgets['reference_bytes']):
        return examples, fallback
    ids = encoded(list(dict.fromkeys([*groups[0],*groups[1]])))
    if signal=='face':
        rows = db.execute('SELECT id,embedding FROM observations WHERE id IN (SELECT value FROM json_each(?))',(ids,))
    else:
        rows = db.execute(f"SELECT observation,{expression('embedding')} FROM person_evidence e "
                          "WHERE contract=? AND status='ready' AND observation IN (SELECT value FROM json_each(?))",(contract(config),ids))
    bank = {k:np.frombuffer(raw,dtype='<f4') for k,raw in rows if raw}
    for a in groups[0]:
        check_cancelled()
        if a not in bank:
            continue
        for b in groups[1]:
            if b in bank and abs(float(np.dot(bank[a],bank[b]))-minimum)<1e-6:
                return (a,b), 'Pair at the reported minimum'
    return examples, fallback


def hybrid_panel(db, config, ids, proof, component):
    panels, settings = [], config.data['sapiensid']['matching']
    for signal, key, guards, floor in [('face','observations','guard_observations','face_floor'),
                                       ('person','body_witness_observations','guard_person_observations','person_threshold')]:
        banks = proof.get(key) or (proof.get('person_observations') if signal=='person' else None) or ((),())
        inherited = proof.get(guards) or ((),())
        groups = [list(dict.fromkeys([*a,*b])) for a,b in zip(banks,inherited)]
        minimum = (proof.get(signal) or {}).get('minimum')
        pair, label = _signal_pair(db,config,groups,signal,minimum)
        if proof['left'] != ids[0]:
            pair = pair[::-1]
        panels.append(dict(signal=signal, ids=pair, label=label, minimum=minimum, floor=settings[floor]))
    return dict(panels=panels, rival=(component or {}).get('nearest_rival'),
                margin=settings['ambiguity_margin'], blockers=proof.get('blockers',()))


def projection_image(config, scope, address):
    from PIL import Image
    from .cold_payloads import get
    from .models import namespace
    with closing(open_readonly(config.database)) as db:
        row=db.execute(f"SELECT {expression('crop')},s.scene,o.cue FROM person_evidence e JOIN observations o "
                       "ON o.id=e.observation JOIN scans s ON s.id=o.scan WHERE e.observation=? AND e.contract=? "
                       "AND s.scope=? AND s.namespace=? AND e.status='ready'",(address,contract(config),scope,namespace(config))).fetchone()
        if row is None:
            return None, 'Body evidence unavailable'
        blob = get(db,row[0])
    caption = f'Scene {row[1]} · cue {row[2]} · body'
    if blob:
        try:
            with Image.open(BytesIO(blob)) as image:
                return image.convert('RGB'), caption
        except (OSError,ValueError):
            return None, caption+' · unreadable crop'
    return None, caption+' · body crop not retained'
