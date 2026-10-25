"""Completed VTT rankings, conditional on exact current local dependencies."""
import hashlib
import json
import math
import sqlite3
import uuid

from .config import digest
from .database import encoded
from .clustering import matching_policy
from .profiling import count
from .visibility import cached as visible,required as needs_visible
from .expression import cached as expressive,required as needs_expression
from .portrait_admission import extra_measurements,reason as anchor_reason


class Ranking:
    def __init__(self,db,config,scope,target,namespace,context,policy_sha,enabled=True):
        self.db,self.config,self.scope,self.target=db,config,scope,target
        self.namespace,self.context,self.policy=namespace,context,policy_sha
        self.limit=config.data['execution']['portrait_ranking_cache_bytes']
        self.enabled=enabled and self.limit>0
        self.key='vtt-ranking-cache:'+digest([scope,target['person'],policy_sha,
            sorted(target['scans'],key=lambda s:(s['scene'],s['scan'])),target.get('matching',{})])
        self.hit=False

    def scan_inputs(self,scan):
        from .dependencies import cached
        from .rebuild_cache import _hash_row
        policy=matching_policy(self.config)
        def compute():
            h=hashlib.sha256()
            for row in self.db.execute('SELECT o.id,o.scan,o.cue,o.seconds,o.source,o.rect,o.geometry,o.quality,'
                'o.dimensions,o.embedding,s.input_sha,s.policy_sha,s.status FROM observations o '
                'JOIN scans s ON s.id=o.scan JOIN members m ON m.observation=o.id '
                'WHERE m.person=? AND o.scan=? AND o.rejected IS NULL ORDER BY o.id',(self.target['person'],scan)):
                _hash_row(h,row)
            for row in self.db.execute('SELECT id,scan,person,status,payload FROM candidates '
                'WHERE policy=? AND scan=? ORDER BY id',(policy,scan)):_hash_row(h,row)
            return h.hexdigest()
        return cached(self.db,'portrait-inputs',digest([self.scope,self.namespace,self.target['person'],scan]),
                      policy,compute,dependency='scan',dependency_key=scan)

    def inputs(self,*,check_retry=False):
        from .rebuild_cache import _hash_row
        h=hashlib.sha256()
        _hash_row(h,[self.scope,self.namespace,self.context,self.policy,self.target['person'],self.target['performer'],
                     encoded(self.target.get('matching',{}))])
        selected=sorted(self.target['scans'],key=lambda s:(s['scene'],s['scan']))
        for start in range(0,len(selected),256):
            ids=encoded([s['scan'] for s in selected[start:start+256]])
            for item in selected[start:start+256]:
                _hash_row(h,[item['scan'],self.scan_inputs(item['scan'])])
            for row in self.db.execute('SELECT id,metadata FROM portraits WHERE scope=? AND person=? '
                "AND json_valid(metadata) AND json_extract(metadata,'$.policy_sha')=? "
                "AND json_extract(metadata,'$.scan') IN (SELECT value FROM json_each(?)) ORDER BY id",
                (self.scope,self.target['person'],self.policy,ids)):
                if check_retry:
                    metadata=json.loads(row['metadata'])
                    if (metadata.get('quality_errors') or metadata.get('source_warning') or metadata.get('fallback')
                            or metadata.get('source_type')=='recognition_crop'):
                        return None
                    if not metadata.get('reason'):
                        quality=metadata.get('quality',{})
                        policy=self.config.data['portraits']
                        if ((needs_visible(policy) or extra_measurements(self.config)) and not visible(quality)
                                or needs_expression(policy) and not expressive(quality)
                                or anchor_reason(quality,self.config)
                                or any(v.get('source_type')=='recognition_crop'
                                       for v in quality.get('measurement_basis',{}).values())):
                            return None
                _hash_row(h,row)
        for row in self.db.execute('SELECT a.source,a.target FROM identity_merges a JOIN people p ON p.id=a.source '
            'WHERE p.scope=? AND p.namespace=? ORDER BY a.source',(self.scope,self.namespace)):
            _hash_row(h,row)
        return h.hexdigest()

    def load(self):
        if not self.enabled:
            return None
        row=self.db.execute('SELECT value FROM metadata WHERE key=?',(self.key,)).fetchone()
        if row is None or len(row[0])>min(1048576,self.limit):
            return None
        try:
            document=json.loads(row[0]);checksum=document.pop('checksum')
            if document.get('format')!=1 or digest(document)!=checksum:
                return None
            ranks=document['ranks']
            if not isinstance(ranks,list) or len({rank[4] for rank in ranks})!=len(ranks):
                return None
            for rank in ranks:
                if (len(rank)!=5 or any(type(rank[i]) not in (int,float) or not math.isfinite(rank[i]) for i in (0,1,3))
                        or not isinstance(rank[2],str) or str(uuid.UUID(rank[4]))!=rank[4]):
                    return None
            if document.get('inputs')!=self.inputs(check_retry=True):
                return None
        except (ValueError,TypeError,KeyError,IndexError,AttributeError,sqlite3.Error):
            return None
        self.hit=True
        count('vtt_complete_ranking_hits')
        return [tuple(rank) for rank in ranks]

    def save(self,ranks):
        if not self.enabled or self.hit:
            return
        inputs=self.inputs(check_retry=True)
        if inputs is None:
            return
        document=dict(format=1,inputs=inputs,ranks=ranks)
        document['checksum']=digest(document)
        payload=encoded(document)
        if len(payload.encode())+len(self.key)+256>min(1048576,self.limit):
            return  # Never truncate ranking or spill a working set.
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(self.key,payload))
            total=0
            for key,size in self.db.execute("SELECT key,length(CAST(value AS BLOB)) FROM metadata "
                "WHERE key LIKE 'vtt-ranking-cache:%' ORDER BY rowid DESC"):
                total+=size+len(key)+256
                if total>self.limit:self.db.execute('DELETE FROM metadata WHERE key=?',(key,))
