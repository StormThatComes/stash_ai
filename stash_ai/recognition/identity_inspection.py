"""Paged, offline reads for the desktop identity inspector."""
from contextlib import closing
from io import BytesIO
import json

from PIL import Image

from . import admission_report
from . import assignment_explanation as explain
from .body_identity import available as bodies_available, policy as body_policy
from .body_observations import contract
from .clustering import matching_policy
from .cold_payloads import get as cold_crop
from .fragmentation_report import _open_readonly
from .identity_report import _assignment, _decisions, _select
from .inspection_detachment import suggest
from .inspection_members import query as member_query
from .inspection_overlays import read as overlay_geometry
from .models import namespace

TABLES = {'face': ('observations', 'members'), 'body': ('body_observations', 'body_members')}
PAGE_SIZE = 80


class IdentityReader:
    def __init__(self, config, *, identity_id=None, performer_id=None, scope=None):
        self.config = config
        self.selector = namespace(config), identity_id, performer_id, scope
        self.face_policy = matching_policy(config)
        self.body_policy = body_policy(config)

    def page(self, *, scene=None, cue=None, observation=None, offset=0, accepted_only=False, kind=None, state=None):
        """Page metadata, optionally restricted to one kind before pagination.

        A pasted observation ID uses the observation primary key.
        """
        if kind is not None and kind not in TABLES:
            raise ValueError('Observation kind must be face or body.')
        with closing(_open_readonly(self.config.database)) as db:
            _, person = _select(db, *self.selector)
            kinds = [k for k in TABLES if (kind is None or kind == k) and (k == 'face' or bodies_available(db))]
            if not kinds:
                return [], False
            sql, args = member_query(person, self.face_policy, kinds, filters=(scene,cue,observation), accepted_only=accepted_only)
            if state is not None:
                sql += ' WHERE state=?'
                args.append(state)
            sql += ' ORDER BY CAST(scene AS INTEGER),scene,cue,kind,id LIMIT ? OFFSET ?'
            rows = [dict(row) for row in db.execute(sql, [*args, PAGE_SIZE+1, offset])]
            return rows[:PAGE_SIZE], len(rows) > PAGE_SIZE

    def states(self, kind):
        """Count accepted member roles across all scenes, only on kind expansion."""
        if kind not in TABLES:
            raise ValueError('Observation kind must be face or body.')
        with closing(_open_readonly(self.config.database)) as db:
            _, person = _select(db, *self.selector)
            if kind == 'body' and not bodies_available(db):
                return []
            sql, args = member_query(person, self.face_policy, [kind], filters=(None,None,None), accepted_only=True)
            return [dict(row) for row in db.execute('SELECT state,COUNT(*) AS count FROM ('+sql+') GROUP BY state ORDER BY state', args)]

    def detail(self, kind, address):
        with closing(_open_readonly(self.config.database)) as db:
            _, person = _select(db, *self.selector)
            observations, members = TABLES[kind]
            row = db.execute(f'SELECT m.*,s.scene,s.policy_sha,s.status AS scan_status,'
                             f'o.cue,o.seconds,o.scan,o.quality,o.geometry,o.rejected FROM {members} m '
                             f'JOIN {observations} o ON o.id=m.observation JOIN scans s ON s.id=o.scan '
                             'WHERE m.observation=? AND m.person=? AND s.scope=? AND s.namespace=?',
                             (address, person['id'], person['scope'], person['namespace'])).fetchone()
            if row is None:
                raise ValueError('This observation is no longer a member. Refresh the list.')
            # Resolve aliases by ID. Do not load every relative's merge proof just to show one crop.
            family = {r[0] for r in db.execute('WITH RECURSIVE family(id) AS (SELECT ? UNION '
                'SELECT m.source FROM identity_merges m JOIN family f ON m.target=f.id '
                'JOIN people p ON p.id=m.source WHERE p.scope=? AND p.namespace=?) SELECT id FROM family',
                (person['id'], person['scope'], person['namespace']))}
            merges = RelatedMerges(db, family)
            policies = admission_report.Policies(db, self.config, person['scope'], person['namespace'])
            quality = json.loads(row['quality'])
            contract_current = True
            if kind == 'face':
                decisions = _decisions(db, person, family, self.face_policy, observation=address).get(address, [])
                trace = explain.face(decisions, _assignment(decisions), merges, person['id'])
                admission = admission_report.face(quality, json.loads(row['geometry']), self.config, policies, row['policy_sha'])
            else:
                body = db.execute('SELECT face,contract FROM body_observations WHERE id=?', (address,)).fetchone()
                admission = admission_report.person(quality, self.config, policies, body['contract'], independent=True)
                contract_current = body['contract'] == contract(self.config)
                trace = explain.body(json.loads(row['proof']), linked_face=body['face'], role=row['role'],
                    status=row['status'], policy=row['policy'], current=row['policy']==self.body_policy,
                    merges=merges, target=person['id'])
            # Keep merge evidence separately from the sample so shared proofs stay shared in reports.
            for step in trace['decisions']:
                for edge in step['merges']:
                    if edge['proof_location'] == 'identity_merges':
                        edge['evidence'] = merges[edge['source']]['evidence']
                    elif kind == 'face':
                        decision = next(d for d in decisions if d['policy']==step['policy'] and d['supports_membership'])
                        evidence = decision['recorded'].get(edge['proof_location'])
                        edge['evidence'] = (evidence[edge['proof_index']]['evidence']
                                            if 'proof_index' in edge else evidence)
            associations = []
            if kind == 'face' and bodies_available(db):
                associations = [dict(kind='body', observation=r[0], label='Linked body in this cue')
                                for r in db.execute('SELECT id FROM body_observations WHERE scan=? AND face=? ORDER BY id',
                                                    (row['scan'], address))]
            return dict(trace=trace, associations=associations, image=_image(db, person, kind, address),
                        geometry=overlay_geometry(db, self.config, person, kind, address),
                        suggestions=suggest(self.config, kind, trace, admission), admission=admission,
                        rejection=row['rejected'], scan_status=row['scan_status'], contract_current=contract_current,
                        scene=row['scene'], cue=row['cue'], seconds=row['seconds'], scope=person['scope'])

    def reference(self, kind, address):
        with closing(_open_readonly(self.config.database)) as db:
            _, person = _select(db, *self.selector)
            return _image(db, person, kind, address)

    def preview(self, kind, address):
        with closing(_open_readonly(self.config.database)) as db:
            _, person = _select(db, *self.selector)
            return dict(image=_image(db, person, kind, address),
                        geometry=overlay_geometry(db, self.config, person, kind, address))


class RelatedMerges(dict):
    def __init__(self, db, family):
        super().__init__()
        self.db, self.family = db, family

    def __contains__(self, source):
        if source not in self.family:
            return False
        if not super().__contains__(source):
            row = self.db.execute('SELECT target,evidence FROM identity_merges WHERE source=?', (source,)).fetchone()
            if row is None or row['target'] not in self.family:
                return False
            self[source] = dict(target=row['target'], evidence=json.loads(row['evidence']))
        return True


def _image(db, person, kind, address):
    table = TABLES[kind][0]
    row = db.execute(f'SELECT o.crop,s.scene,o.cue FROM {table} o JOIN scans s ON s.id=o.scan '
                     'WHERE o.id=? AND s.scope=? AND s.namespace=?',
                     (address, person['scope'], person['namespace'])).fetchone()
    if row is None:
        return None, 'Observation unavailable in this scope and model.'
    caption = f"Scene {row['scene']} · cue {row['cue']} · {kind}"
    try:
        blob = cold_crop(db, row['crop'])
        if blob is None:
            return None, caption + ' · Crop not retained'
        with Image.open(BytesIO(blob)) as source:
            return source.convert('RGB'), caption
    except (OSError, Image.DecompressionBombError, ValueError):
        return None, caption + ' · Retained crop unreadable'
