"""Ordered, replayable witness selection for one owner in configured workspace SQLite."""
import json
from .database import encoded
from .rooted_witnesses import select_ordered, retrieves_independent_witness


class Tracks:
    def __init__(self, db, witness):
        self.db, self.witness = db, witness
        db.execute('CREATE TABLE IF NOT EXISTS witnesses(id TEXT PRIMARY KEY, scan TEXT, shot TEXT, '
                   'overlap REAL, points INTEGER, sharpness REAL, payload TEXT, active INTEGER)')
        db.execute('CREATE INDEX IF NOT EXISTS witness_scan ON witnesses(scan,id)')
        db.execute('CREATE INDEX IF NOT EXISTS witness_order ON witnesses(overlap,points DESC,sharpness DESC,id DESC)')

    def add(self, row, shot):
        q = json.loads(row['quality']).get('quality', {})
        self.db.execute('INSERT INTO witnesses VALUES(?,?,?,?,?,?,?,1)', (
            row['id'], row['scan'], encoded(shot), q.get('overlap',0), q.get('keypoints',0),
            q.get('sharpness',0), encoded(row)))

    def modes(self, scan, settings, scores, rank, shot_of):
        scope = '1' if scan is None else 'scan=?'
        args = () if scan is None else (scan,)
        shot = 'scan' if scan is None else 'shot'
        order = 'overlap,points DESC,sharpness DESC,id DESC'
        self.db.execute('UPDATE witnesses SET active=1 WHERE '+scope,args)
        query = ' FROM witnesses WHERE active=1 AND '+scope
        def decode(raw):
            row = self.witness(json.loads(raw))
            return (*row['roots'], row)
        class Ordered:
            def __iter__(ordered):
                for raw, in self.db.execute('SELECT payload'+query+' ORDER BY '+order,args):
                    yield decode(raw)
        index = 0
        while True:
            count = self.db.execute('SELECT COUNT(DISTINCT '+shot+')'+query,args).fetchone()[0]
            if not count:
                return
            positions = sorted({round(i*(count-1)/7) for i in range(8)}) if count>8 else list(range(count))
            seed_sql = ('WITH ranked AS (SELECT *,row_number() OVER (PARTITION BY '+shot+' ORDER BY '+order+
                        ') AS n'+query+'), seeds AS (SELECT payload,row_number() OVER (ORDER BY '+order+
                        ')-1 AS pos FROM ranked WHERE n=1) SELECT payload FROM seeds WHERE pos IN '
                        '(SELECT value FROM json_each(?)) ORDER BY pos')
            seeds = [decode(raw) for raw, in self.db.execute(seed_sql,(*args,encoded(positions)))]
            kept = select_ordered(Ordered(), seeds, settings, scores, rank, shot_of)
            if not kept or index and len(kept)<max(2,settings['attach_min_support']):
                break
            if index<3 or retrieves_independent_witness(kept,settings,scores):
                yield kept
            index += 1
            used = {root for a,b,_ in kept for root in (a,b)}
            # Stable keyset pages allow updating the active set without modifying a live cursor.
            last = ''
            while rows := self.db.execute('SELECT id,payload'+query+' AND id>? ORDER BY id LIMIT 128',
                                          (*args,last)).fetchall():
                retire = []
                for last, raw in rows:
                    a,b,row = decode(raw)
                    covered = a in used or b in used
                    if not covered:
                        values = [scores.face_similarity(row['_vector'],w['_vector']) for _,_,w in kept]
                        covered = min(values)>=settings['cohesion_threshold'] and sum(values)/len(values)>=settings['match_threshold']
                    if covered:
                        retire.append((last,))
                self.db.executemany('UPDATE witnesses SET active=0 WHERE id=?',retire)
