"""Conservative disk postings from changed view evidence to retained scenes."""
import base64
import json
import math
import numpy as np

from .config import digest
from .database import encoded
from .dependencies import version


class ReverseIndex:
    """Buckets enclose every query projection. Exact gallery checks still decide."""
    def __init__(self, db, scope, namespace, stamp, pending):
        self.db = db
        self.prefix = 'rear-index:'+digest([scope,namespace])+':'
        self.previous = self.read('generation',{})
        signature = digest(['v2',stamp,pending.radius,pending.floor])
        if self.previous.get('signature')!=signature:
            db.execute('DELETE FROM metadata WHERE key GLOB ?',(self.prefix+'*',))
            self.previous = {}
        versions = dict(zip(pending.banks,(pending.gallery.owner_at(i) for i in range(len(pending.banks)))))
        self.state = dict(signature=signature,banks=versions,scans={},
            owners={p:[v,version(db,'owner',p),p in getattr(pending.search,'unavailable',())]
                    for p,v in pending.search.mean_versions.items()})
        saved = self.read('axes',None)
        if saved is None:
            axes = []
            count_banks = len(pending.banks)
            for i in dict.fromkeys(round(step*(count_banks-1)/4) for step in range(5)):
                value = np.asarray(pending.gallery.banks[i][2][0]['_vector'],dtype=float).copy()
                for axis in axes:
                    value -= (axis@value)*axis
                norm = np.linalg.norm(value)
                if norm>.001:
                    axes.append(value/norm)
                if len(axes)==3:
                    break
            saved = [base64.b64encode(axis.astype('<f4').tobytes()).decode() for axis in axes]
            self.write('axes',saved)
        self.axes = []
        for item in saved:
            axis = np.frombuffer(base64.b64decode(item),dtype='<f4').astype(float)
            self.axes.append(axis/np.linalg.norm(axis))

    def read(self,k,default):
        row = self.db.execute('SELECT value FROM metadata WHERE key=?',(self.prefix+k,)).fetchone()
        return json.loads(row[0]) if row else default

    def write(self,k,value):
        self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(self.prefix+k,encoded(value)))

    def buckets(self,center,radius):
        for i,axis in enumerate(self.axes):
            projection = float(axis@center)
            # Unit queries project into [-1,1]; outer bins include rounding.
            lo = max(0,min(32,math.floor((projection-radius+1)*16)))
            hi = max(0,min(32,math.floor((projection+radius+1)*16)))
            yield {f'b{i}_{n:02d}' for n in range(lo,hi+1)}

    def bins(self,center,radius):
        return set().union(*self.buckets(center,radius))

    def nearby(self,center,radius):
        candidates = None
        for tokens in self.buckets(center,radius):
            hits = self.members(tokens)
            candidates = hits if candidates is None else candidates & hits
            if not candidates:
                break
        return candidates or set()

    def members(self,tokens):
        result = set()
        for token in tokens:
            start = self.prefix+token+':'
            result.update(row[0] for row in self.db.execute(
                'SELECT value FROM metadata WHERE key>=? AND key<?',(start,start[:-1]+';')))
        return result

    def store(self,scan,entry,banks,pending):
        owners = set(entry['local']) | set(entry.get('rivals',()))
        owners.update(pending.gallery.owner_at(i) for i in banks)
        tokens = {'o'+digest(owner) for owner in owners}
        for pivot,radius in entry['balls']:
            center = np.frombuffer(base64.b64decode(pivot),dtype='<f4')
            tokens.update(self.bins(center,radius+.003))
        self.post(scan,tokens)

    def post(self,scan,tokens):
        key = 'post:'+scan
        old = self.read(key,None)
        if old is not None and set(old)==tokens:
            return
        old = set(old or ())
        def address(token):
            return self.prefix+token+':'+scan
        self.db.executemany('DELETE FROM metadata WHERE key=?',((address(t),) for t in old-tokens))
        self.db.executemany('INSERT OR REPLACE INTO metadata VALUES (?,?)',((address(t),scan) for t in tokens-old))
        self.write(key,sorted(tokens))

    def nominate(self,pending,changed):
        if not self.previous or changed is None or not self.axes:
            return None
        old,new = self.previous['banks'],self.state['banks']
        added = new.keys()-old.keys()
        old_owners,owners = self.previous['owners'],self.state['owners']
        touched = {old[key] for key in old.keys()-new.keys()}
        touched.update(p for p in old_owners.keys()|owners.keys() if old_owners.get(p)!=owners.get(p))
        if len(added)>64 or len(touched)>256:
            return None
        result = self.members({'o'+digest(owner) for owner in touched})
        for i,bank in enumerate(pending.banks):
            if bank in added:
                for row in pending.gallery.banks[i][2]:
                    result.update(self.nearby(row['_vector'],pending.radius+.003))
        for center,norm in changed:
            # ||q-c||² = ||q||²+||c||²-2 q.c. Include the scalar veto tolerance.
            radius = math.sqrt(max(0.,1.+norm*norm-2*(pending.floor-.004)))+.003
            result.update(self.nearby(center,radius))
        return result

    def finish(self):
        for scan in self.previous.get('scans',{}).keys()-self.state['scans'].keys():
            self.post(scan,set())
            self.db.execute('DELETE FROM metadata WHERE key=?',(self.prefix+'post:'+scan,))
        if self.state!=self.previous:
            self.write('generation',self.state)
