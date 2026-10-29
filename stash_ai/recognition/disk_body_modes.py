"""Complete hybrid witnesses on disk, with bounded vector working storage."""
from pathlib import Path
import pickle
import sqlite3
import tempfile
import numpy as np

from .independent_evidence import Votes,unit


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
            raise ValueError('Body appearance construction exceeds sapiensid.execution.max_witness_comparisons.')
        result=float(np.dot(left,right));self.count+=1
        self.progress.advance()
        return result


class Records:
    """Repeatable SQL spool; only bounded selected rows return to the catalogue."""
    def __init__(self):
        self.directory=tempfile.TemporaryDirectory(prefix='stash-hybrid-witnesses-')
        self.path=Path(self.directory.name)
        self.db=sqlite3.connect(self.path/'records.sqlite')
        self.db.execute('PRAGMA cache_size=-512')
        self.db.execute('PRAGMA journal_mode=OFF')
        self.db.execute('CREATE TABLE records (ordinal INTEGER PRIMARY KEY,id TEXT UNIQUE,payload BLOB)')
        self.db.execute('CREATE TABLE features (cluster INTEGER,embedding BLOB,PRIMARY KEY(cluster,embedding))')
        self.db.execute('CREATE TABLE scored (cluster INTEGER,unit BLOB,PRIMARY KEY(cluster,unit))')

    def add(self,row):
        self.db.execute('INSERT INTO records(id,payload) VALUES (?,?)',(row['id'],pickle.dumps(dict(row),protocol=5)))

    def rows(self):
        for row in self.db.execute('SELECT payload FROM records ORDER BY ordinal'):
            yield pickle.loads(row[0])

    def selected(self,ids):
        wanted=set(ids)
        return [r for r in self.rows() if r['id'] in wanted]

    def modes(self,config,budget=None):
        if budget is not None:return self._modes(config,budget)
        from .progress import work
        with work('Build complete body appearances',unit='comparison') as progress:
            return self._modes(config,ComparisonBudget(config,progress))

    def _modes(self,config,budget):
        from .hybrid_references import body_vector
        from .memory_size import size
        threshold=config.data['sapiensid']['matching']['person_threshold']
        minimum=config.data['sapiensid']['matching']['min_support']
        votes=[];counts=[];support=Votes();charge=0;seen=0
        for row in self.rows():
            value=body_vector(row);blob=value.tobytes();compatible=[]
            for cluster in range(len(votes)):
                if self.db.execute('SELECT 1 FROM features WHERE cluster=? AND embedding=?',(cluster,blob)).fetchone():
                    compatible.append(cluster);continue
                if all(budget.score(value,np.frombuffer(r[0],dtype='<f4'))>=threshold
                       for r in self.db.execute('SELECT embedding FROM features WHERE cluster=?',(cluster,))):
                    compatible.append(cluster)
            if len(compatible)==1:cluster=compatible[0]
            else:
                cluster=len(votes);votes.append(Votes());counts.append(0)
            self.db.execute('INSERT OR IGNORE INTO features VALUES (?,?)',(cluster,blob))
            with (self.path/(str(cluster)+'.bin')).open('ab') as target:target.write(blob)
            fresh=self.db.execute('INSERT OR IGNORE INTO scored VALUES (?,?)',
                                  (cluster,pickle.dumps(unit(row),protocol=5))).rowcount
            with (self.path/(str(cluster)+'.weights')).open('ab') as target:target.write(bytes([int(bool(fresh))]))
            votes[cluster].add(row);support.add(row);counts[cluster]+=1
            # Bound vote metadata as well as vector tiles and SQLite page cache.
            seen+=1
            if seen%128==1:
                charge=size([vars(v) for v in votes]+[vars(support)])
            else:charge+=2048
            charge+=len(row['quality'])
            if charge+2*1024**2>config.data['sapiensid']['execution']['reference_bytes']:
                raise ValueError('SapiensID reference_bytes budget exceeded by mode vote metadata.')
        if any(len(v)<minimum for v in votes):return (),0
        return tuple(Matrix(self.path/(str(i)+'.bin'),n,self.directory,self.path/(str(i)+'.weights'))
                     for i,n in enumerate(counts)),len(support)

    def close(self,*,keep=False):
        self.db.close()
        if not keep:self.directory.cleanup()
