"""Scratch-backed exact radius search over independently certified view banks."""
from collections.abc import Sequence
import base64
import hashlib
import json

import numpy as np

from .config import digest
from .dependencies import version
from .profiling import count
from .database import encoded
from .rear_gallery import Gallery
from .resource_errors import ResourceLimit


class Banks(Sequence):
    def __init__(self, db, feature=None):
        self.db, self.feature = db, feature
        self.owners = []

    def __len__(self):
        return len(self.owners)

    def __getitem__(self, index):
        owner, source = self.owners[index]
        rows = []
        for metadata, blob in self.db.execute(
                'SELECT metadata,embedding FROM gallery WHERE bank=? ORDER BY ordinal',(index,)):
            row = json.loads(metadata)
            row['_vector'] = self.feature(row['id']) if blob is None else np.frombuffer(blob,dtype='<f4')
            rows.append(row)
        return owner, source, rows


class DiskGallery(Gallery):
    def __init__(self, db, banks, limit, *, feature=None, cache=None, source=None):
        self.db = db
        db.execute('PRAGMA cache_size=-'+str(max(16,min(2048,limit//8192))))
        db.execute('CREATE TABLE gallery(bank INTEGER,ordinal INTEGER,metadata TEXT,embedding BLOB,'
                   'PRIMARY KEY(bank,ordinal)) WITHOUT ROWID')
        db.execute('CREATE TABLE bounds(bank INTEGER PRIMARY KEY,block INTEGER,center BLOB,radius REAL)')
        db.execute('CREATE INDEX bounds_block ON bounds(block,bank)')
        db.execute('CREATE TABLE blocks(block INTEGER PRIMARY KEY,center BLOB,radius REAL)')
        self.banks = Banks(db,feature)
        self.versions = []
        block = []
        used = 0
        block_size = max(1,min(64,limit//(8*4096*8)))
        for index,(owner,source_name,rows) in enumerate(banks):
            used += len(owner)+len(source_name)+320
            if used + 8*len(rows)*16384 > limit:
                raise ResourceLimit('Perspective gallery directory exceeds sapiensid.execution.reference_bytes.')
            self.banks.owners.append((owner,source_name))
            # Scratch rows are addresses into immutable retained evidence.
            # The bounded durable cache holds numerical bounds for unchanged banks.
            clocks = {scan:version(source,'raw-body',scan) for scan in {r['scan'] for r in rows}} if cache else {}
            key = digest(['rear-bounds-v1',owner,source_name,clocks,[dict(r) for r in rows]]) if cache else None
            kind = 'body-appearance-rear-bounds'
            stored = cache.get(kind,key) if cache else None
            stamp = hashlib.sha256(encoded([owner,source_name]).encode())
            vectors = []
            for ordinal, row in enumerate(rows):
                payload = {k:v for k,v in row.items() if k!='_vector'}
                vector = row['_vector'] if stored is None or feature is None else None
                db.execute('INSERT INTO gallery VALUES (?,?,?,?)',(index,ordinal,encoded(payload),
                           None if feature else vector.astype('<f4',copy=False).tobytes()))
                if stored is None:
                    vectors.append(vector.astype('float64'))
                    stamp.update(encoded([row['id'],row['roots'],row['quality'],row.get('perspective','back')]).encode())
                    stamp.update(vector.tobytes())
            if stored is None:
                center = np.mean(vectors,axis=0)
                radius = max(np.linalg.norm(v-center) for v in vectors)
                checksum = stamp.hexdigest()
                count('rear_gallery_rebuilt_bounds')
                if cache:
                    cache.put(kind,key,[checksum,base64.b64encode(center.astype('<f8').tobytes()).decode(),radius])
            else:
                checksum,raw_center,radius = stored
                center = np.frombuffer(base64.b64decode(raw_center),dtype='<f8')
                count('rear_gallery_reused_bounds')
            self.versions.append(checksum)
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
                for blob,oid in self.db.execute("SELECT embedding,json_extract(metadata,'$.id') FROM gallery WHERE bank=?",(bank,)):
                    vector = self.banks.feature(oid) if blob is None else np.frombuffer(blob,dtype='<f4')
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
