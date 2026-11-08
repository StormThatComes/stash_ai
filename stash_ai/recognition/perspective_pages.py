"""Scratch-backed exact radius search over independently certified view banks."""
from collections.abc import Sequence
import hashlib
import json

import numpy as np

from .database import encoded
from .rear_gallery import Gallery
from .resource_errors import ResourceLimit


class Banks(Sequence):
    def __init__(self, db):
        self.db = db
        self.owners = []

    def __len__(self):
        return len(self.owners)

    def __getitem__(self, index):
        owner, source = self.owners[index]
        rows = []
        for metadata, blob in self.db.execute(
                'SELECT metadata,embedding FROM gallery WHERE bank=? ORDER BY ordinal',(index,)):
            rows.append(dict(json.loads(metadata), _vector=np.frombuffer(blob,dtype='<f4')))
        return owner, source, rows


class DiskGallery(Gallery):
    def __init__(self, db, banks, limit):
        self.db = db
        db.execute('PRAGMA cache_size=-'+str(max(16,min(2048,limit//8192))))
        db.execute('CREATE TABLE gallery(bank INTEGER,ordinal INTEGER,metadata TEXT,embedding BLOB,'
                   'PRIMARY KEY(bank,ordinal)) WITHOUT ROWID')
        db.execute('CREATE TABLE bounds(bank INTEGER PRIMARY KEY,block INTEGER,center BLOB,radius REAL)')
        db.execute('CREATE INDEX bounds_block ON bounds(block,bank)')
        db.execute('CREATE TABLE blocks(block INTEGER PRIMARY KEY,center BLOB,radius REAL)')
        self.banks = Banks(db)
        self.versions = []
        block = []
        used = 0
        block_size = max(1,min(64,limit//(8*4096*8)))
        for index,(owner,source,rows) in enumerate(banks):
            used += len(owner)+len(source)+320
            if used + 8*len(rows)*16384 > limit:
                raise ResourceLimit('Perspective gallery directory exceeds sapiensid.execution.reference_bytes.')
            self.banks.owners.append((owner,source))
            stamp = hashlib.sha256(encoded([owner,source]).encode())
            vectors = []
            for ordinal, row in enumerate(rows):
                vector = row['_vector']
                vectors.append(vector.astype('float64'))
                db.execute('INSERT INTO gallery VALUES (?,?,?,?)',
                           (index,ordinal,encoded({k:v for k,v in row.items() if k!='_vector'}),vector.astype('<f4',copy=False).tobytes()))
                stamp.update(encoded([row['id'],row['roots'],row['quality'],row.get('perspective','back')]).encode())
                stamp.update(vector.tobytes())
            self.versions.append(stamp.hexdigest())
            center = np.mean(vectors,axis=0)
            radius = max(np.linalg.norm(v-center) for v in vectors)
            db.execute('INSERT INTO bounds VALUES (?,?,?,?)',
                       (index,index//block_size,center.astype('<f8').tobytes(),radius))
            block.append((center,radius))
            if len(block)==block_size:
                self._finish_block(index//block_size,block)
                block.clear()
        if block:
            self._finish_block((len(self.banks)-1)//block_size,block)
        self.tree = self if len(self.banks) else None

    def _finish_block(self,index,block):
        center = np.mean([c for c,_ in block],axis=0)
        radius = max(np.linalg.norm(c-center)+r for c,r in block)
        self.db.execute('INSERT INTO blocks VALUES (?,?,?)',(index,center.astype('<f8').tobytes(),radius))

    def owner_at(self,index):
        return self.banks.owners[index][0]

    def nearby(self,query,radius):
        return self.nearby_many([query],[radius])[0]

    def nearby_many(self,queries,radii):
        queries = [np.asarray(q,dtype='float64') for q in queries]
        results = [set() for _ in queries]
        def inside(blob, extent, indices):
            pivot = np.frombuffer(blob,dtype='<f8')
            return [i for i in indices if np.linalg.norm(queries[i]-pivot)<=radii[i]+extent+1e-9]
        for block,center,extent in self.db.execute('SELECT block,center,radius FROM blocks'):
            possible = inside(center,extent,range(len(queries)))
            if not possible:
                continue
            for bank,pivot,spread in self.db.execute('SELECT bank,center,radius FROM bounds WHERE block=?',(block,)):
                active = inside(pivot,spread,possible)
                if not active:
                    continue
                for (blob,) in self.db.execute('SELECT embedding FROM gallery WHERE bank=?',(bank,)):
                    vector = np.frombuffer(blob,dtype='<f4')
                    for i in tuple(active):
                        if np.linalg.norm(queries[i]-vector)<=radii[i]:
                            results[i].add(bank)
                            active.remove(i)
                    if not active:
                        break
        return results

    def rank_many(self,requests):
        if not requests:
            return []
        queries = [r[0] for r in requests]
        radii = [self.search_radius(settings) for _,_,settings,_,_ in requests]
        candidates = self.nearby_many(queries,radii)
        ranked = [[] for _ in queries]
        for index in sorted(set().union(*candidates)):
            bank = self.banks[index]
            for i,indices in enumerate(candidates):
                if index in indices:
                    ranked[i].append(self.score_bank(bank,*requests[i]))
        return [self.ordered(rows) for rows in ranked]
