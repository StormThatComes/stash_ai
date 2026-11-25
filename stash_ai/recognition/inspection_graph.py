"""Bounded typed graph of retained decisions, never a reconstruction of matching.

All reads share one SQLite snapshot. Decision nodes carry aggregate scores; an
edge to a witness does not assign that score to the individual comparison.
"""
from contextlib import closing
import hashlib
import json
from . import assignment_explanation as explain
from .identity_inspection import TABLES, RelatedMerges
from .identity_report import _select, _decisions, _assignment
from .inspection_members import query
from .inspection_runtime import open_readonly, stamp, check_cancelled
from .body_identity import available
from .reference_roles import strict_sql, verified_sql

MAX_NODES = 500
PAGE_SIZE = 100


def key(kind, address):
    return kind + ':' + address


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:24]


def build(reader, *, scene=None, kind=None, state=None, offset=0, limit=PAGE_SIZE):
    if kind not in (None, 'face', 'body') or not 1 <= limit <= PAGE_SIZE or offset < 0:
        raise ValueError('Invalid graph page or kind.')
    revision = stamp(reader.config.database)
    nodes, edges = {}, {}
    omitted = 0
    def node(value):
        nonlocal omitted
        if value['id'] not in nodes and len(nodes) >= MAX_NODES:
            omitted += 1
            return False
        nodes.setdefault(value['id'], value)
        return True
    def edge(source, target, relation, provenance):
        if source in nodes and target in nodes:
            edges[source,target,relation] = dict(source=source,target=target,relation=relation,provenance=provenance)
    with closing(open_readonly(reader.config.database)) as db:
        requested, person = _select(db, *reader.selector)
        kinds = [k for k in TABLES if (kind is None or k==kind) and (k=='face' or available(db))]
        if not kinds:
            return dict(nodes=[],edges=[],total=0,more=False,offset=offset,omitted=0,person=person,revision=revision)
        sql, args = query(person, reader.face_policy, kinds, filters=(scene,None,None),
                          accepted_only=True,config=reader.config)
        if state:
            sql += ' WHERE state=?'; args.append(state)
        total = db.execute('SELECT COUNT(*) FROM ('+sql+')', args).fetchone()[0]
        rows = [dict(r) for r in db.execute(sql+' ORDER BY CAST(scene AS INTEGER),scene,cue,kind,id LIMIT ? OFFSET ?',
                                          [*args,limit,offset])]
        family = {r[0] for r in db.execute('WITH RECURSIVE family(id) AS (SELECT ? UNION '
            'SELECT m.source FROM identity_merges m JOIN family f ON m.target=f.id '
            'JOIN people p ON p.id=m.source WHERE p.scope=? AND p.namespace=?) SELECT id FROM family',
            (person['id'],person['scope'],person['namespace']))}
        merges = RelatedMerges(db, family)
        for row in rows:
            node(dict(row,id=key(row['kind'],row['id']),observation=row['id'],type='observation',
                      current_member=True,authority='unknown',provenance='current_membership'))
        def reference(ref):
            identifier = key(ref['kind'],ref['observation'])
            if identifier in nodes: return identifier
            table = TABLES[ref['kind']][0]
            found = db.execute(f'SELECT o.id,s.scene,o.cue FROM {table} o JOIN scans s ON s.id=o.scan '
                'WHERE o.id=? AND s.scope=? AND s.namespace=?',
                (ref['observation'],person['scope'],person['namespace'])).fetchone()
            node(dict(id=identifier,type='observation',kind=ref['kind'],observation=ref['observation'],
                      scene=found['scene'] if found else None,cue=found['cue'] if found else None,
                      state='reference',authority='unknown',current_member=False,
                      availability='retained_context' if found else 'unavailable_in_scope',
                      provenance='recorded_reference'))
            return identifier
        for row in rows:
            check_cancelled()
            target = key(row['kind'],row['id'])
            if row['kind']=='face':
                decisions = _decisions(db,person,family,reader.face_policy,observation=row['id']).get(row['id'],[])
                trace = explain.face(decisions,_assignment(decisions),merges,person['id'])
                authority = db.execute('SELECT '+strict_sql()+','+verified_sql()+' FROM observations o '
                    'JOIN members m ON m.observation=o.id WHERE o.id=?',(row['id'],)).fetchone()
                # Legacy SQL admits missing decisions. The graph never invents authority from it.
                nodes[target]['authority'] = ('strict_route' if authority[0] else 'direct_auxiliary' if authority[1]
                                               else 'attachment_only') if decisions else 'unknown'
            else:
                member = db.execute('SELECT m.*,o.face FROM body_members m JOIN body_observations o '
                                    'ON o.id=m.observation WHERE o.id=?',(row['id'],)).fetchone()
                proof = json.loads(member['proof'])
                trace = explain.body(proof,linked_face=member['face'],role=member['role'],status=member['status'],
                    policy=member['policy'],current=member['policy']==reader.body_policy,merges=merges,target=person['id'])
                nodes[target]['authority'] = ('scene_anchor' if proof.get('reference_scope')=='scene' else 'body_anchor') if row['role']=='anchor' else 'attachment_only'
            nodes[target]['evidence_status'] = trace['evidence_status']
            nodes[target]['decisions'] = len(trace['decisions'])
            for decision in trace['decisions']:
                raw = (next((d['recorded'].get(explain.FACE_PROOF.get(decision['reason']), {}) for d in decisions
                             if d['policy']==decision['policy'] and d['supports_membership']), {})
                       if row['kind']=='face' else proof)
                support_count = raw.get('support_count',raw.get('support',raw.get('independent_support')))
                if type(support_count) is not int: support_count=None
                provenance = trace['evidence_status']
                did = 'decision:'+fingerprint([person['scope'],person['namespace'],target,decision])
                if not node(dict(id=did,type='decision',reason=decision['reason'],policy=decision['policy'],
                    metrics=decision['metrics'],provenance=provenance,merges=decision['merges'],
                    reference_completeness='not_recorded',reference_selected=decision.get('reference_selected'))):
                    continue
                edge(did,target,'assigned',provenance)
                refs = decision['references']
                # A View here is the recorded witness set of this decision, not a
                # freshly rebuilt catalogue or a claim of complete View membership.
                view_refs = [r for r in refs if r['label'] in ('View medoid','View support','Support ids')]
                view = None
                if len({(r['kind'],r['observation']) for r in view_refs if r['label']=='View medoid'})==1:
                    view = 'view:'+fingerprint([person['scope'],person['namespace'],decision['policy'],
                        decision['reason'],view_refs,decision['metrics']])
                    if node(dict(id=view,type='view',provenance=provenance,policy=decision['policy'],
                        support_count=support_count,
                        retained_examples=len({(r['kind'],r['observation']) for r in view_refs if r['label']!='View medoid'}),
                        completeness='partial_or_unknown',label='Recorded View evidence')):
                        edge(view,did,'view_match',provenance)
                    else: view = None
                for ref in refs:
                    check_cancelled()
                    rid = reference(ref)
                    label = ref['label']
                    relation = ('association' if label=='Linked face' else
                                'rival' if 'rival' in label.lower() or 'peer' in label.lower() else
                                'track' if 'track' in label.lower() else 'reference')
                    if view and ref in view_refs:
                        edge(rid,view,'medoid' if label=='View medoid' else 'view_support',provenance)
                    else:
                        edge(rid,did,relation,provenance)
        result = dict(nodes=list(nodes.values()),edges=list(edges.values()),total=total,offset=offset,
                      more=offset+len(rows)<total,omitted=omitted,person=person,requested=requested,revision=revision,
                      complete=False,coverage='Retained references only; absent links do not prove independence.')
    if revision != stamp(reader.config.database):
        raise ValueError('Recognition changed during graph loading. Refresh the graph.')
    return result
