"""Complete hybrid witnesses on disk, with bounded vector working storage."""
from .resource_errors import ResourceLimit
from pathlib import Path
import pickle
import sqlite3
import tempfile
import numpy as np

from .independent_evidence import Votes,unit,stamp


class Matrix:
    """Pickle only an immutable file address; workers never copy the full bank."""
    def __init__(self,path,count,owner=None,weights_path=None):
        self.path=str(path);self.shape=(count,4096);self.owner=owner
        self.weights_path=str(weights_path) if weights_path is not None else None
        self.nbytes=count*4096*4

    def __len__(self):return self.shape[0]

    def __getitem__(self,index):
        if isinstance(index,int):
            start=index if index>=0 else len(self)+index
            if not 0<=start<len(self):raise IndexError(index)
            return self[start:start+1][0]
        start,stop,step=index.indices(len(self))
        if step!=1:raise ValueError('Body disk banks require contiguous slices.')
        with open(self.path,'rb') as source:
            source.seek(start*4096*4)
            values=np.fromfile(source,dtype='<f4',count=max(0,stop-start)*4096)
        if values.size!=max(0,stop-start)*4096:raise ValueError('Incomplete body witness spool.')
        return values.reshape(-1,4096)

    def weights(self,index):
        start,stop,step=index.indices(len(self))
        if step!=1:raise ValueError('Body weights require contiguous slices.')
        if self.weights_path is None:return np.ones(max(0,stop-start),dtype='float64')
        with open(self.weights_path,'rb') as source:
            source.seek(start);values=np.fromfile(source,dtype='uint8',count=max(0,stop-start))
        if values.size!=max(0,stop-start):raise ValueError('Incomplete body weight spool.')
        return values.astype('float64')

    def __getstate__(self):return self.path,self.shape,self.weights_path
    def __setstate__(self,state):
        self.path,self.shape,self.weights_path=state;self.owner=None;self.nbytes=self.shape[0]*4096*4


def resident_bytes(value):
    return 256 if isinstance(value,Matrix) else value.nbytes


class ComparisonBudget:
    """One exact-comparison budget across all owners in a catalogue."""
    def __init__(self,config,progress):
        self.limit=config.data['sapiensid']['execution']['max_witness_comparisons']
        self.count=0;self.progress=progress

    def score(self,left,right):
        if self.count>=self.limit:
            raise ResourceLimit('Body appearance construction exceeds sapiensid.execution.max_witness_comparisons.')
        result=float(np.dot(left,right));self.count+=1
        self.progress.advance()
        return result


class Records:
    """Repeatable SQL spool; only bounded selected rows return to the catalogue."""
    def __init__(self):
        from .scratch_space import directory,Connection
        self.directory=directory(prefix='stash-hybrid-witnesses-')
        self.path=Path(self.directory.name)
        self.db=sqlite3.connect(self.path/'records.sqlite',factory=Connection)
        self.db.execute('PRAGMA cache_size=-512')
        self.db.execute('PRAGMA journal_mode=OFF')
        self.db.execute('CREATE TABLE records (ordinal INTEGER PRIMARY KEY,id TEXT UNIQUE,payload BLOB)')
        self.db.execute('CREATE TABLE bounds(cluster INTEGER PRIMARY KEY,center BLOB,radius REAL,max_norm REAL)')
        self.db.execute('CREATE TABLE features (cluster INTEGER,embedding BLOB,PRIMARY KEY(cluster,embedding))')
        self.db.execute('CREATE TABLE scored (ordinal INTEGER PRIMARY KEY,cluster INTEGER,stamp BLOB,unit BLOB)')

    def add(self,row):
        self.db.execute('INSERT INTO records(id,payload) VALUES (?,?)',(row['id'],pickle.dumps(dict(row),protocol=5)))

    def rows(self):
        for row in self.db.execute('SELECT payload FROM records ORDER BY ordinal'):
            yield pickle.loads(row[0])

    def selected(self,ids):
        from .database import encoded
        return [pickle.loads(r[0]) for r in self.db.execute(
            'SELECT payload FROM records WHERE id IN (SELECT value FROM json_each(?)) ORDER BY ordinal',
            (encoded(list(ids)),))]

    def modes(self,config,budget=None):
        if budget is not None:return self._modes(config,budget)
        from .progress import work
        with work('Build complete body appearances',unit='comparison') as progress:
            return self._modes(config,ComparisonBudget(config,progress))

    def _modes(self,config,budget):
        from collections import OrderedDict
        self.writers=OrderedDict()
        try:return self._build_modes(config,budget)
        finally:
            from stash_ai.runtime.cleanup import cleanup
            cleanup([('close body bank stream',f.close) for f in self.writers.values()])
            self.writers.clear()

    def _write(self,cluster,blob):
        target=self.writers.get(cluster)
        if target is None:
            if len(self.writers)>=8:self.writers.popitem(last=False)[1].close()
            target=(self.path/(str(cluster)+'.bin')).open('ab')
            self.writers[cluster]=target
        self.writers.move_to_end(cluster)
        from .scratch_space import grow
        grow(self.path/(str(cluster)+'.bin'),len(blob))
        target.write(blob)

    def _build_modes(self,config,budget):
        from .hybrid_references import body_vector
        from .memory_size import size
        threshold=config.data['sapiensid']['matching']['person_threshold']
        minimum=config.data['sapiensid']['matching']['min_support']
        votes=[];counts=[];support=Votes();charge=0;seen=0
        for row in self.rows():
            value=body_vector(row);blob=value.tobytes();compatible=[]
            wide=value.astype('float64');norm=float(np.linalg.norm(wide))
            for cluster in range(len(votes)):
                if self.db.execute('SELECT 1 FROM features WHERE cluster=? AND embedding=?',(cluster,blob)).fetchone():
                    compatible.append(cluster);continue
                center,radius,max_norm=self.db.execute('SELECT center,radius,max_norm FROM bounds WHERE cluster=?',(cluster,)).fetchone()
                center=np.frombuffer(center,dtype='<f8')
                slack=8*len(value)*np.finfo(np.float32).eps*max(1.,norm*max_norm)
                lower=float(np.dot(wide,center))-norm*radius-slack
                if np.isfinite(lower) and lower>=threshold:
                    from .profiling import count
                    count('body_mode_acceptance_certificates');compatible.append(cluster);continue
                if all(budget.score(value,np.frombuffer(r[0],dtype='<f4'))>=threshold
                       for r in self.db.execute('SELECT embedding FROM features WHERE cluster=?',(cluster,))):
                    compatible.append(cluster)
            if len(compatible)==1:cluster=compatible[0]
            else:
                cluster=len(votes);votes.append(Votes());counts.append(0)
                self.db.execute('INSERT INTO bounds VALUES (?,?,?,?)',(cluster,wide.astype('<f8').tobytes(),0.,norm))
            center,radius,max_norm=self.db.execute('SELECT center,radius,max_norm FROM bounds WHERE cluster=?',(cluster,)).fetchone()
            radius=max(radius,float(np.linalg.norm(wide-np.frombuffer(center,dtype='<f8')))+1e-12)
            self.db.execute('UPDATE bounds SET radius=?,max_norm=? WHERE cluster=?',(radius,max(max_norm,norm),cluster))
            self.db.execute('INSERT OR IGNORE INTO features VALUES (?,?)',(cluster,blob))
            self._write(cluster,blob)
            self.db.execute('INSERT INTO scored(cluster,stamp,unit) VALUES (?,?,?)',
                            (cluster,pickle.dumps(stamp(row),protocol=5),pickle.dumps(unit(row),protocol=5)))
            votes[cluster].add(row);support.add(row);counts[cluster]+=1
            # Bound vote metadata as well as vector tiles and SQLite page cache.
            seen+=1
            if seen%128==1:
                charge=size([vars(v) for v in votes]+[vars(support)])
            else:charge+=2048
            charge+=len(row['quality'])
            if charge+2*1024**2>config.data['sapiensid']['execution']['reference_bytes']:
                raise ResourceLimit('SapiensID reference_bytes budget exceeded by mode vote metadata.')
        if any(len(v)<minimum for v in votes):return (),0
        # Use the same maximum cue/pixel matching as support. Keep every vector
        # for minimum similarity and contradiction checks, including zero weights.
        for cluster,vote in enumerate(votes):
            used=set()
            from .scratch_space import grow
            grow(self.path/(str(cluster)+'.weights'),counts[cluster])
            with (self.path/(str(cluster)+'.weights')).open('wb') as target:
                for timestamp,pixel in self.db.execute('SELECT stamp,unit FROM scored WHERE cluster=? ORDER BY ordinal',(cluster,)):
                    timestamp,pixel=pickle.loads(timestamp),pickle.loads(pixel)
                    fresh=pixel not in used and vote.owners.get(pixel)==timestamp
                    if fresh:used.add(pixel)
                    target.write(bytes([int(fresh)]))
        return tuple(Matrix(self.path/(str(i)+'.bin'),n,self.directory,self.path/(str(i)+'.weights'))
                     for i,n in enumerate(counts)),len(support)

    def close(self,*,keep=False):
        self.db.close()
        (self.path/'records.sqlite').unlink(missing_ok=True)
        if not keep:
            from .scratch_space import release_directory
            self.directory.cleanup();release_directory(self.path)
