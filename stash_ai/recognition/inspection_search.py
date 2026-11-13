"""Read-only searches across the complete scene-inspector membership scope."""
from contextlib import closing
import json

from .identity_inspection import PAGE_SIZE, IdentityReader
from .inspection_members import query
from .inspection_runtime import open_readonly, cached
from .models import namespace
from .reference_roles import observation_sql
from .scene_inspection import UnassignedReader, _unassigned_query
from .body_observations import available
from .clustering import matching_policy
from .body_identity import policy as body_policy

SORTS = {'Scene / cue':'CAST(scene AS INTEGER),scene,cue,kind,id',
         'Lowest lead':'margin IS NULL,margin,kind,id',
         'Lowest sharpness':'sharpness IS NULL,sharpness,kind,id',
         'State':'state,CAST(scene AS INTEGER),scene,cue,kind,id'}


class SearchReader:
    def __init__(self, scene_reader, filters=None, scan=None):
        self.source, self.config = scene_reader, scene_reader.config
        self.scan=scan
        self.filters = dict(filters or {})
        self.selector = ('search',scene_reader.scene,scene_reader.scope,json.dumps(self.filters,sort_keys=True))

    @cached
    def page(self, *, offset=0, **unused):
        catalogue=self.source.catalogue();scan=catalogue['scan'];self.scan=scan
        face_policy,body_current=matching_policy(self.config),body_policy(self.config)
        with closing(open_readonly(self.config.database)) as db:
            has_body=available(db)
            pieces=[];args=[]
            for group in catalogue['groups']:
                person=dict(id=group['identity'],scope=scan['scope'],namespace=scan['namespace'])
                sql,params=query(person,face_policy,['face','body'] if has_body else ['face'],filters=(None,None,None),accepted_only=True)
                pieces.append('SELECT r.*,? AS identity,? AS performer FROM ('+sql+') r')
                args.extend((group['identity'],group['performer'],*params))
            for kind in ('face','body') if has_body else ('face',):
                sql=_unassigned_query(kind)
                pieces.append('SELECT kind,id,scene,cue,seconds,status,role,state,NULL AS identity,NULL AS performer FROM ('+sql+')')
                args.extend((scan['scene'],scan['id']))
            # Restrict accepted proofs to the actual owner/its merged predecessors.
            family=('WITH RECURSIVE family(id) AS (SELECT r.identity UNION SELECT m.source FROM identity_merges m '
                    'JOIN family f ON m.target=f.id JOIN people p ON p.id=m.source '
                    'WHERE p.scope=? AND p.namespace=?) SELECT id FROM family')
            proof=("(SELECT json_object('reason',json_extract(c.payload,'$.reason'),'policy',c.policy,"
                   "'lead',CASE WHEN json_extract(c.payload,'$.reason') IN ('known_identity','candidate_group','auxiliary_identity','same_scene_identity') "
                   "THEN json_extract(c.payload,'$.scores[0]')-json_extract(c.payload,'$.scores[1]') END) "
                   f'FROM candidates c WHERE {observation_sql("c")}=r.id AND c.scan=o.scan '
                   "AND (r.identity IS NULL OR (c.status='accepted' AND c.person IN ("+family+'))) '
                   'ORDER BY (c.policy=?) DESC,c.rowid DESC LIMIT 1)')
            fields=("SELECT r.*,o.rejected,o.crop IS NULL AS missing_crop,json_extract(o.quality,'$.sharpness') AS sharpness,"+proof+
                    " AS proof FROM rows r JOIN observations o ON o.id=r.id WHERE r.kind='face'")
            args.extend((scan['scope'],scan['namespace'],face_policy))
            if has_body:
                fields+=(" UNION ALL SELECT r.*,o.rejected,o.crop IS NULL,json_extract(o.quality,'$.quality.sharpness'),"
                         "json_object('reason',json_extract(m.proof,'$.reason'),'policy',m.policy,'lead',"
                         "CASE WHEN json_extract(m.proof,'$.runner')>=0 THEN json_extract(m.proof,'$.score')-json_extract(m.proof,'$.runner') END) "
                         "FROM rows r JOIN body_observations o ON o.id=r.id LEFT JOIN body_members m ON m.observation=o.id WHERE r.kind='body'")
            sql=('WITH rows AS ('+' UNION ALL '.join(pieces)+'), details AS ('+fields+'), result AS ('
                 "SELECT *,json_extract(proof,'$.reason') AS reason,json_extract(proof,'$.lead') AS margin,"
                 "CASE WHEN json_extract(proof,'$.policy') IS NULL THEN 'missing' WHEN json_extract(proof,'$.policy')="
                 "CASE WHEN kind='face' THEN ? ELSE ? END THEN 'current' ELSE 'historical' END AS policy_state FROM details) SELECT * FROM result WHERE 1=1")
            args.extend((face_policy,body_current))
            for name,column in (('scene','scene'),('kind','kind'),('state','state'),('policy','policy_state'),('performer','performer')):
                value=self.filters.get(name)
                if value and value!='All':sql+=' AND '+column+'=?';args.append(value)
            for name,column in (('text','id'),('reason','reason')):
                value=self.filters.get(name)
                if value:sql+=' AND instr(lower(COALESCE('+column+",'')),lower(?))>0";args.append(value)
            if self.filters.get('missing_crop'):sql+=' AND missing_crop=1'
            if self.filters.get('max_margin') not in (None,''):
                maximum=float(self.filters['max_margin'])
                if not 0<=maximum<=2:raise ValueError('Maximum lead must be between 0 and 2.')
                sql+=' AND margin<=?';args.append(maximum)
            order=SORTS.get(self.filters.get('sort','Scene / cue'))
            if order is None:raise ValueError('Unknown sort order.')
            found=[dict(r) for r in db.execute(sql+' ORDER BY '+order+' LIMIT ? OFFSET ?',(*args,PAGE_SIZE+1,offset))]
        for row in found:row.pop('proof',None)
        return found[:PAGE_SIZE],len(found)>PAGE_SIZE

    def owner(self, row):
        scan=self.scan or self.source.catalogue()['scan']
        return (IdentityReader(self.config,identity_id=row['identity'],scope=scan['scope']) if row['identity'] else UnassignedReader(self.config,scan))
