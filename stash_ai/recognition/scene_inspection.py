"""Scene entry point for retained memberships and unassigned evidence."""
from contextlib import closing
import json

from . import admission_report, assignment_explanation as explain
from .body_identity import available as bodies_available, policy as body_policy
from .body_observations import contract
from .clustering import matching_policy
from .decision_contexts import unpack
from .decision_evidence import recorded
from .inspection_runtime import open_readonly as _open_readonly, cached
from .identity_inspection import IdentityReader, PAGE_SIZE, TABLES, _image
from .inspection_advice import suggest
from .inspection_overlays import read as overlay_geometry
from .models import namespace
from .inspection_policy import selection
from .scanning import scan_identity
from .reference_roles import observation_sql
from .scan_heads import predicate as current_scan


def _accepted(kind):
    return "m.person IS NOT NULL" + (" AND m.status='accepted'" if kind == 'body' else '')


def _reason(payload):
    return (f"COALESCE(CASE WHEN json_valid({payload}) THEN CASE WHEN "
            f"json_type({payload},'$.reason')='text' THEN NULLIF(trim(json_extract({payload},'$.reason')),'') "
            "END END,'unknown_reason')")


def _unassigned_query(kind, config, scan):
    """Share membership and displayed-state rules between counts and pages."""
    observations, members = TABLES[kind]
    status = ("COALESCE(m.status,'unassigned')" if kind == 'body' else
              "CASE WHEN c.status='withheld' THEN 'withheld' ELSE 'unassigned' END")
    role = "COALESCE(m.role,'unknown')" if kind == 'body' else "'unknown'"
    state = f"COALESCE(NULLIF(o.rejected,''),CASE WHEN {status}='accepted' THEN {role} ELSE {status} END)"
    current, args = selection(config, kind)
    evidence = ''
    decision_args = []
    if kind == 'face':
        # A retained historical attempt must not turn an unmatched face into a
        # current withheld result. The latest current decision wins exactly once.
        evidence = ('LEFT JOIN candidates c ON c.rowid=(SELECT fc.rowid FROM candidates fc '
                    f'WHERE {observation_sql("fc")}=o.id AND fc.scan=o.scan AND fc.policy=? '
                    'ORDER BY fc.rowid DESC LIMIT 1) ')
        decision_args = [matching_policy(config)]
    reason = _reason('m.proof' if kind == 'body' else 'c.payload')
    return (f"SELECT '{kind}' AS kind,o.id,s.scene,o.cue,o.seconds,"
            f'{status} AS status,{role} AS role,o.rejected,{state} AS state,{reason} AS reason FROM {observations} o '
            f'JOIN scans s ON s.id=o.scan LEFT JOIN {members} m ON m.observation=o.id '
            + evidence + f'WHERE o.scan=? AND NOT ({_accepted(kind)}) AND ' + current, [*decision_args, scan['id'], *args])


class SceneReader:
    def __init__(self, config, scene, *, scope=None):
        self.config, self.scene, self.scope = config, str(scene), scope
        self.namespace = namespace(config)

    @cached
    def catalogue(self):
        with closing(_open_readonly(self.config.database)) as db:
            scans = db.execute('SELECT s.id,s.scene,s.scope,s.namespace,s.policy_sha,s.status FROM scans s WHERE s.scene=? AND s.namespace=? '
                'AND (? IS NULL OR s.scope=?) AND ' + current_scan(),
                (self.scene, self.namespace, self.scope, self.scope)).fetchall()
            if not scans:
                raise ValueError('No retained scan for this scene, model namespace and scope.')
            if len(scans) != 1:
                raise ValueError('Scene ID occurs in multiple Stash scopes. Start with --scope.')
            scan = dict(scans[0])
            if scan['policy_sha'] != scan_identity(self.config, self.namespace):
                raise ValueError('No current scan matches this YAML. Run recognition for this scene, then refresh.')
            groups, unassigned_groups = {}, []
            for kind, (observations, members) in TABLES.items():
                if kind == 'body' and not bodies_available(db):
                    continue
                current, args = selection(self.config, kind)
                for row in db.execute(f'SELECT p.id,p.performer,COUNT(*) AS count FROM {observations} o '
                    f'JOIN scans s ON s.id=o.scan JOIN {members} m ON m.observation=o.id JOIN people p ON p.id=m.person '
                    f'WHERE o.scan=? AND {_accepted(kind)} AND p.scope=? AND p.namespace=? '
                    'AND '+current+' GROUP BY p.id,p.performer', (scan['id'], scan['scope'], self.namespace, *args)):
                    group = groups.setdefault(row['id'], dict(identity=row['id'], performer=row['performer'], count=0))
                    group['count'] += row['count']
                query, args = _unassigned_query(kind, self.config, scan)
                states = {}
                for row in db.execute('SELECT state,reason,COUNT(*) AS count FROM (' + query +
                                      ') GROUP BY state,reason ORDER BY state,COUNT(*) DESC,reason', args):
                    group = states.setdefault(row['state'], dict(kind=kind, state=row['state'], count=0))
                    group['count'] += row['count']
                    if row['state'] == 'withheld':
                        group.setdefault('reasons', []).append(dict(reason=row['reason'], count=row['count']))
                unassigned_groups.extend(states.values())
            # Count only scene-local metadata. All-scene member pages are loaded on expansion.
            return dict(scan=scan, unassigned=sum(g['count'] for g in unassigned_groups),
                        unassigned_groups=unassigned_groups,
                        groups=sorted(groups.values(), key=lambda g: (not bool(g['performer']), g['performer'] or '', g['identity'])))

    def identity_reader(self, identity, scope):
        return IdentityReader(self.config, identity_id=identity, scope=scope)


class UnassignedReader:
    """No face membership or no accepted body membership, regardless of role.

    Provisional body tracks stay here. Accepted identities without a Stash ID
    have their own tree nodes, preserving their existing membership evidence.
    """
    def __init__(self, config, scan):
        self.config, self.scan = config, scan
        self.face_policy, self.body_policy = matching_policy(config), body_policy(config)

    @cached
    def page(self, *, offset=0, kind=None, state=None, reason=None, after=None, observation=None):
        """Filter metadata by kind and displayed state before applying pagination."""
        if kind is not None and kind not in TABLES:
            raise ValueError('Observation kind must be face or body.')
        with closing(_open_readonly(self.config.database)) as db:
            queries, args = [], []
            for sample_kind in TABLES:
                if kind is not None and sample_kind != kind:
                    continue
                if sample_kind == 'body' and not bodies_available(db):
                    continue
                query, params = _unassigned_query(sample_kind, self.config, self.scan)
                queries.append(query)
                args.extend(params)
            if not queries:
                return [], False
            sql = 'SELECT * FROM ('+' UNION ALL '.join(queries)+')'
            filters = []
            for column, value in (('state', state), ('reason', reason)):
                if value is not None:
                    filters.append(column+'=?')
                    args.append(value)
            if after is not None:
                filters.append('(cue,kind,id)>(?,?,?)')
                args.extend((after['cue'],after['kind'],after['id']))
            if observation is not None:
                filters.append('id=?')
                args.append(observation)
            if filters:
                sql += ' WHERE '+' AND '.join(filters)
            rows = [dict(r) for r in db.execute(sql+' ORDER BY cue,kind,id LIMIT ? OFFSET ?',
                                               (*args, PAGE_SIZE+1, offset))]
            return rows[:PAGE_SIZE], len(rows)>PAGE_SIZE

    def detail(self, kind, address, *, pixels=True):
        with closing(_open_readonly(self.config.database)) as db:
            observations, members = TABLES[kind]
            current, args = selection(self.config, kind)
            row = db.execute(f'SELECT o.id,o.scan,o.cue,o.seconds,o.quality,o.geometry,o.rejected{',o.face,o.contract' if kind == 'body' else ''} FROM {observations} o JOIN scans s ON s.id=o.scan LEFT JOIN {members} m ON m.observation=o.id '
                f'WHERE o.id=? AND o.scan=? AND NOT ({_accepted(kind)}) AND '+current, (address, self.scan['id'], *args)).fetchone()
            if row is None:
                raise ValueError('This observation is no longer unassigned in the current results for this YAML. Refresh the tree.')
            policies = admission_report.Policies(db, self.config, self.scan['scope'], self.scan['namespace'])
            quality = json.loads(row['quality'])
            decisions, variants, associations = [], [], []
            if kind == 'face':
                admission = admission_report.face(quality, json.loads(row['geometry']), self.config, policies, self.scan['policy_sha'])
                for candidate in db.execute('SELECT c.* FROM candidates c WHERE c.scan=? AND '+observation_sql('c')+
                                            '=? ORDER BY c.rowid DESC', (row['scan'], address)):
                    proof = recorded(unpack(db, candidate['payload']))
                    item = _decision(candidate, proof, self.face_policy)
                    decisions.append(item)
                    # Retain the path prefix: unsuccessful probes are not the final decision's score.
                    metrics, refs = {}, []
                    for key in (*dict.fromkeys(explain.FACE_PROOF.values()), 'anchor_view_probe', 'auxiliary_view_probe'):
                        evidence = proof.get(key)
                        if isinstance(evidence, dict):
                            metrics.update({key+'.'+k: v for k,v in evidence.items() if k in explain.MEASUREMENTS})
                            refs.extend(explain.references(evidence, 'face'))
                    if proof.get('scores'):
                        metrics['recorded_scores'] = proof['scores']
                    refs.extend(explain.references(dict(scoring_references=proof.get('compared_references')), 'face'))
                    variants.append(_variant(item, metrics, refs))
                if bodies_available(db):
                    associations = [dict(kind='body', observation=r[0], label='Linked body in this cue') for r in
                        db.execute('SELECT id FROM body_observations WHERE scan=? AND face=? AND contract=? ORDER BY id', (row['scan'], address, contract(self.config)))]
            else:
                from .face_landmark_fallback import retained_body_face
                admission = admission_report.person(quality, self.config, policies, row['contract'], independent=True,
                    face_evidence=retained_body_face(db, address, self.config.data['sapiensid']['identities']))
                member = db.execute('SELECT * FROM body_members WHERE observation=?', (address,)).fetchone()
                if member:
                    proof = json.loads(member['proof'])
                    item = _decision(member, proof, self.body_policy)
                    decisions.append(item)
                    variants.append(_variant(item, {k: proof[k] for k in explain.MEASUREMENTS if proof.get(k) is not None},
                                             explain.references(proof, 'body', row['face'])))
                elif row['face']:
                    associations.append(dict(kind='face', observation=row['face'], label='Linked face'))
            trace = dict(status='unassigned', role='no accepted membership', decisions=variants,
                         evidence_status='retained_decisions' if decisions else 'no_matching_decision')
            return dict(trace=trace, recorded_decisions=decisions, admission=admission, rejection=row['rejected'],
                        geometry=overlay_geometry(db, self.config, self.scan, kind, address),
                        suggestions=suggest(self.config, kind, admission, decisions, row['rejected']),
                        associations=associations, image=_image(db, self.scan, kind, address) if pixels else (None, 'Loading crop…'), scene=self.scan['scene'],
                        cue=row['cue'], seconds=row['seconds'], scan_status=self.scan['status'],
                        contract_current=kind=='face' or row['contract']==contract(self.config), scope=self.scan['scope'])

    def reference(self, kind, address):
        with closing(_open_readonly(self.config.database)) as db:
            return _image(db, self.scan, kind, address)

    def preview(self, kind, address):
        with closing(_open_readonly(self.config.database)) as db:
            return dict(image=_image(db, self.scan, kind, address),
                        geometry=overlay_geometry(db, self.config, self.scan, kind, address))


def _decision(row, proof, policy):
    return dict(status=row['status'], policy=row['policy'], current=row['policy']==policy,
                person=row['person'], recorded=proof,
                explanation='This retained attempt does not establish a current accepted membership.')


def _variant(item, metrics, refs):
    if metrics.get('runner') == -1 and metrics.get('rival') is None:
        metrics.pop('runner')
    return dict(reason=item['recorded'].get('reason', 'unknown'), summary='Recorded attempt: '+item['status'],
                policy=item['policy'], metrics=metrics, references=refs, merges=[], merge_source=None)
