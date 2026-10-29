"""Bounded centroid nomination with exact retained-witness acceptance and rivals."""
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
import sqlite3
import tempfile

import numpy as np

from .body_identity import vector
from .memory_size import Reservation,size
from .progress import tracking

# Conservative float32 dot-product error envelope for 4096 normalized terms.
# It only adds nominations; all gates and near-tied rivals use the old np.dot
# reduction, including its acceptance-side rounding at configured thresholds.
ROUNDING = .001


class Evidence:
    def __init__(self,db,config,cache_bytes,stats):
        self.db=db;self.limit=config.data['sapiensid']['execution']['max_witness_comparisons']
        self.stats=stats;self.cache=OrderedDict();self.used=0;self.cache_bytes=cache_bytes
        self.scores=OrderedDict()

    def feature(self,row,kind='body'):
        key=(kind,row['id'])
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        if kind=='body':
            raw=self.db.execute('SELECT embedding FROM body_observations WHERE id=?',(row['id'],)).fetchone()
            value=vector(raw)
        else:
            from .clustering import vector as face_vector
            raw=self.db.execute('SELECT embedding,dimensions FROM observations WHERE id=?',(row['id'],)).fetchone()
            value=face_vector(raw)
        self.stats['embedding_reads']+=1
        charge=value.nbytes+512
        while self.cache and self.used+charge>self.cache_bytes:
            _,removed=self.cache.popitem(last=False);self.used-=removed.nbytes+512
        if charge<=self.cache_bytes:
            self.cache[key]=value;self.used+=charge
        return value

    def compare(self,left,right,kind='body'):
        key=(kind,tuple(r['id'] for r in left),tuple(r['id'] for r in right))
        if key in self.scores:
            self.stats['score_cache_hits']+=1;self.scores.move_to_end(key);return self.scores[key]
        count=len(left)*len(right)
        charged=self.stats['body_witness_comparisons']+self.stats['face_witness_comparisons']+count
        if charged>self.limit:
            raise ValueError(f'Global body witness comparisons {charged} exceed '
                f'sapiensid.execution.max_witness_comparisons={self.limit}; '
                f"candidate pairs={self.stats['candidate_pairs']}.")
        self.stats[kind+'_witness_comparisons']+=count
        total=0.;minimum=float('inf')
        for a in left:
            first=self.feature(a,kind)
            for b in right:
                second=self.feature(b,kind)
                if len(first)!=len(second):raise ValueError('Incompatible face namespaces in body consolidation.')
                score=float(np.dot(first,second));total+=score;minimum=min(minimum,score)
        result=(total/count,minimum) if count else (-1.,-1.)
        # Keep the cache small even for large inherited veto banks.
        if len(left)+len(right)<=128:
            self.scores[key]=result
            if len(self.scores)>128:self.scores.popitem(last=False)
        return result


class Search:
    """Disk-backed means; only two small tiles and a bounded vector LRU in RAM."""
    def __init__(self,db,config,bank):
        self.bank=bank;self.identities=sorted(bank);self.config=config;self.db=db
        self.stats=dict(identities=len(bank),identity_pairs=len(bank)*(len(bank)-1)//2,
            candidate_pairs=0,body_witness_comparisons=0,face_witness_comparisons=0,
            score_cache_hits=0,embedding_reads=0,centroid_pairs=0)
        self.stack=ExitStack()

    def __enter__(self):
        try:
            limit=self.config.data['sapiensid']['execution']['reference_bytes']
            metadata=size(self.bank)+size(self.identities)
            # Includes fetched SQLite blobs, NumPy copies, product tiles and
            # the scratch SQLite page cache; no resident N-by-4096 allocation.
            self.block=min(64,max(1,(limit-metadata-4*1024**2)//(8*4096*12)))
            working=6*self.block*4096*8+4*1024**2
            if metadata+working+1024**2>limit:
                raise ValueError('Global body lookup exceeds sapiensid.execution.reference_bytes.')
            cache_bytes=min(16*1024**2,(limit-metadata-working-1024**2)//2)
            self.reservation=Reservation(limit)
            self.reservation.set(metadata+working+cache_bytes+1024**2)
            self.stack.callback(self.reservation.set,0)
            self.stats['reference_working_bytes']=metadata+working+cache_bytes+1024**2
            self.evidence=Evidence(self.db,self.config,cache_bytes,self.stats)
            directory=self.stack.enter_context(tempfile.TemporaryDirectory(prefix='stash-body-means-'))
            path=Path(directory)/'means.sqlite'
            self.index=sqlite3.connect(path);self.stack.callback(self.index.close)
            self.index.execute('PRAGMA journal_mode=OFF');self.index.execute('PRAGMA cache_size=-1024')
            self.index.execute('PRAGMA mmap_size=0')
            self.index.execute('CREATE TABLE means (ordinal INTEGER PRIMARY KEY,vector BLOB NOT NULL)')
            with tracking(self.identities,'Index global person means',unit='identity') as people:
                for i,person in enumerate(people):
                    refs=self.bank[person]['refs'];mean=np.zeros(4096,dtype='<f8')
                    for row in refs:mean+=self.evidence.feature(row)
                    if refs:mean/=len(refs)
                    self.index.execute('INSERT INTO means VALUES (?,?)',(i,mean.tobytes()))
            self.index.commit();self.stats['centroid_disk_bytes']=path.stat().st_size
            return self
        except BaseException:
            self.stack.close();raise

    def __exit__(self,*args):
        self.evidence.cache.clear();self.evidence.scores.clear()
        return self.stack.__exit__(*args)

    def tiles(self,active=None):
        count=len(self.identities);block=self.block
        def read(start):
            return np.array([np.frombuffer(r[0],dtype='<f8') for r in self.index.execute(
                'SELECT vector FROM means WHERE ordinal>=? AND ordinal<? ORDER BY ordinal',(start,start+block))])
        with tracking(range(0,count,block),'Compare global person means',unit='block') as starts:
            for i in starts:
                left=read(i)
                for j in range(i,count,block):
                    if active is not None and not (active[i:i+block].any() or active[j:j+block].any()):continue
                    right=left if j==i else read(j)
                    values=left @ right.T
                    self.stats['centroid_pairs']+=(len(left)*(len(left)-1)//2 if i==j else values.size)
                    yield i,j,values

    def nominations(self,focus):
        floor=self.config.data['sapiensid']['global_matching']['match_threshold']
        wanted=np.array([p in focus for p in self.identities])
        for i,j,values in self.tiles():
            rows=np.arange(i,i+values.shape[0]);cols=np.arange(j,j+values.shape[1])
            mask=(rows[:,None]<cols[None,:]) & ((values+ROUNDING>=floor) |
                  (wanted[rows,None] & wanted[None,cols]))
            for a,b in zip(*np.nonzero(mask)):
                self.stats['candidate_pairs']+=1
                limit=self.config.data['sapiensid']['execution']['max_candidate_pairs']
                if self.stats['candidate_pairs']>limit:
                    raise ValueError(f"Global body candidate pairs {self.stats['candidate_pairs']} exceed "
                        f'sapiensid.execution.max_candidate_pairs={limit}; identities={len(self.identities)}.')
                yield self.identities[i+a],self.identities[j+b]

    def nearest(self,lanes,pairs):
        """Exact strongest outsider for each group member, including weak rivals.

        A first tiled pass finds approximate maxima. A second checks every
        witness pair within the conservative rounding envelope of that maximum.
        No top-k truncation or discarded-candidate assumption can hide a rival.
        """
        groups=[np.array([lane.get(p,-1) for p in self.identities]) for lane in lanes]
        active=np.array([any(g[i]>=0 for g in groups) for i in range(len(self.identities))])
        if not active.any():return [{} for _ in lanes]
        maxima=[np.full(len(self.identities),-np.inf) for _ in lanes]
        result=[{} for _ in lanes]
        def rows(i,j,values):
            yield i,j,values
            if i!=j:yield j,i,values.T
        for i,j,values in self.tiles(active):
            for start,other,tile in rows(i,j,values):
                for lane,g in enumerate(groups):
                    for r in np.flatnonzero(g[start:start+len(tile)]>=0):
                        index=start+r;allowed=g[other:other+tile.shape[1]]!=g[index]
                        if allowed.any():maxima[lane][index]=max(maxima[lane][index],float(tile[r,allowed].max()))
        for i,j,values in self.tiles(active):
            for start,other,tile in rows(i,j,values):
                for lane,g in enumerate(groups):
                    for r in np.flatnonzero(g[start:start+len(tile)]>=0):
                        index=start+r;a=self.identities[index]
                        allowed=(g[other:other+tile.shape[1]]!=g[index]) & (tile[r]+2*ROUNDING>=maxima[lane][index])
                        for c in np.flatnonzero(allowed):
                            b=self.identities[other+c];key=tuple(sorted((a,b)))
                            known=pairs.get(key)
                            score=(known['score'] if known is not None else
                                self.evidence.compare(self.bank[key[0]]['refs'],self.bank[key[1]]['refs'])[0])
                            previous=result[lane].get(a)
                            if previous is None or (-score,b)<(-previous['score'],previous['rival']):
                                result[lane][a]=dict(identity=a,rival=b,score=score)
        return result


def exact_pair(a,b,bank,config,evidence):
    first,second=bank[a],bank[b];settings=config.data['sapiensid']['global_matching']
    mean,minimum=evidence.compare(first['refs'],second['refs']);blockers=[]
    if mean<settings['match_threshold']:blockers.append('body_similarity')
    for left in first['cores']:
        for right in second['cores']:
            m,n=evidence.compare(left,right);minimum=min(minimum,n)
            if m<settings['match_threshold'] or n<settings['cohesion_threshold']:
                if 'body_cohesion' not in blockers:blockers.append('body_cohesion')
    if min(first['support'],second['support'])<settings['min_support']:blockers.append('independent_support')
    if first['cues']&second['cues']:blockers.append('person_copresence')
    if first['missing_original'] or second['missing_original']:blockers.append('missing_original_body_witness')
    if first.get('stale_policy') or second.get('stale_policy'):
        blockers.append('historical_body_policy_requires_rebuild')
    face_min=None
    if first['faces'] and second['faces']:
        face_min=evidence.compare(first['faces'],second['faces'],'face')[1]
    if face_min is not None and face_min<config.data['sapiensid']['matching']['face_floor']:blockers.append('face_contradiction')
    return dict(people=[a,b],score=mean,minimum=minimum,face_minimum=face_min,
        support=[first['support'],second['support']],face_evidence='compared' if face_min is not None else 'unavailable',
        blockers=blockers)
