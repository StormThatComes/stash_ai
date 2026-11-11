"""Paged, offline reads for the desktop identity inspector."""
from contextlib import closing
from io import BytesIO
import json

from PIL import Image

from . import assignment_explanation as explain
from .body_identity import available as bodies_available, policy as body_policy
from .clustering import matching_policy
from .cold_payloads import get as cold_crop
from .fragmentation_report import _open_readonly
from .identity_report import _assignment, _decisions, _select
from .models import namespace

TABLES = {'face': ('observations', 'members'), 'body': ('body_observations', 'body_members')}
PAGE_SIZE = 80


class IdentityReader:
    def __init__(self, config, *, identity_id=None, performer_id=None, scope=None):
        self.config = config
        self.selector = namespace(config), identity_id, performer_id, scope
        self.face_policy = matching_policy(config)
        self.body_policy = body_policy(config)

    def page(self, *, scene=None, cue=None, observation=None, offset=0, accepted_only=False):
        """Page metadata only. A pasted observation ID uses the observation primary key."""
        with closing(_open_readonly(self.config.database)) as db:
            _, person = _select(db, *self.selector)
            queries, args = [], []
            for kind, (observations, members) in TABLES.items():
                if kind == 'body' and not bodies_available(db):
                    continue
                state = "m.status" if kind == 'body' else "'accepted'"
                role = "m.role" if kind == 'body' else "'member'"
                sql = (f"SELECT '{kind}' AS kind,o.id,s.scene,o.cue,o.seconds,{state} AS status,{role} AS role "
                       f'FROM {members} m JOIN {observations} o ON o.id=m.observation '
                       'JOIN scans s ON s.id=o.scan WHERE m.person=? AND s.scope=? AND s.namespace=?')
                args.extend((person['id'], person['scope'], person['namespace']))
                if accepted_only and kind == 'body':
                    sql += " AND m.status='accepted'"
                for column, value in (('s.scene', scene), ('o.cue', cue), ('o.id', observation)):
                    if value is not None:
                        sql += f' AND {column}=?'
                        args.append(value)
                queries.append(sql)
            sql = 'SELECT * FROM (' + ' UNION ALL '.join(queries) + ') ORDER BY CAST(scene AS INTEGER),scene,cue,kind,id LIMIT ? OFFSET ?'
            rows = [dict(row) for row in db.execute(sql, [*args, PAGE_SIZE+1, offset])]
            return rows[:PAGE_SIZE], len(rows) > PAGE_SIZE

    def detail(self, kind, address):
        with closing(_open_readonly(self.config.database)) as db:
            _, person = _select(db, *self.selector)
            observations, members = TABLES[kind]
            row = db.execute(f'SELECT m.*,s.scene,o.cue,o.seconds,o.scan FROM {members} m '
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
            if kind == 'face':
                decisions = _decisions(db, person, family, self.face_policy, observation=address).get(address, [])
                trace = explain.face(decisions, _assignment(decisions), merges, person['id'])
            else:
                linked = db.execute('SELECT face FROM body_observations WHERE id=?', (address,)).fetchone()[0]
                trace = explain.body(json.loads(row['proof']), linked_face=linked, role=row['role'],
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
                        scene=row['scene'], cue=row['cue'], seconds=row['seconds'])

    def reference(self, kind, address):
        with closing(_open_readonly(self.config.database)) as db:
            _, person = _select(db, *self.selector)
            return _image(db, person, kind, address)


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
