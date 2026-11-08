"""Two-person VTT workload with recurring actors, outfit changes and close rivals."""
from contextlib import ExitStack, closing
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from stash_ai.recognition.config import digest
from stash_ai.recognition.rear_gallery import Gallery
from stash_ai.recognition.rear_pending import _bank_versions
from stash_ai.recognition.rear_reverse import ReverseIndex
from stash_ai.recognition.scratch_space import connect, scope


def process_io():
    path = Path('/proc/self/io')
    if not path.is_file():
        return {}
    return {k:int(v) for line in path.read_text().splitlines() for k,v in [line.split(':')]
            if k in ('read_bytes','write_bytes')}


def run(scenes,config_path=None):
    if scenes<1:
        raise ValueError('A positive scene count is required.')
    rng = np.random.default_rng(1219)
    actors,looks,angles,dimensions = 48,3,4,4096
    def normalized(rows):
        return (rows/np.linalg.norm(rows,axis=-1,keepdims=True)).astype('<f4')
    base = normalized(rng.normal(size=(actors,dimensions)))
    # Pairs share part of their visual identity and can become credible rivals.
    base[1::2] = normalized(.82*base[::2]+.50*base[1::2])
    outfits = normalized(rng.normal(size=(actors,looks,dimensions)))
    pose = normalized(rng.normal(size=(angles,dimensions)))
    noise = normalized(rng.normal(size=(actors,looks,angles,dimensions)))
    vectors = normalized(base[:,None,None,:]+.33*outfits[:,:,None,:]+.32*pose+.10*noise).reshape(-1,dimensions)
    actor_ids = np.arange(actors)
    def catalogue(changed):
        features = vectors.reshape(actors,looks,angles,dimensions)
        banks = []
        for owner in actor_ids:
            appearance = int(changed and owner in (0,17))
            pid = str(owner)
            banks.append((pid,'external',[dict(id=f'{pid}-{appearance}',roots=[pid+'a',pid+'b'],
                         quality='{}',_vector=features[owner,appearance,0])]))
        gallery = Gallery(banks)
        return SimpleNamespace(gallery=gallery,banks=_bank_versions(gallery),radius=.57,floor=.84,
                               search=SimpleNamespace(mean_versions={}))
    old,new = catalogue(False),catalogue(True)
    def scores(catalogue):
        refs = np.stack([rows[0]['_vector'] for _,_,rows in catalogue.gallery.banks])
        return vectors @ refs.T
    old_scores,new_scores = scores(old),scores(new)
    floor = 1-old.radius**2/2
    old_hits,new_hits = old_scores>=floor,new_scores>=floor
    popularity = np.arange(1,actors+1,dtype=float)**-1.15
    popularity /= popularity.sum()
    # Each positive cue uses two physical tracks and one of four perspectives.
    # Repeated retained view vectors share arithmetic, never ground-truth votes.
    queries = []
    for i in range(scenes):
        first,second = rng.choice(actors,size=2,replace=False,p=popularity)
        if i%4==0:
            second=int(first)^1
        wardrobe=rng.integers(0,looks,size=2)
        schedule = rng.integers(0,angles,size=(210,2))
        addresses = ((np.array([first,second])*looks+wardrobe)*angles+schedule).ravel()
        queries.append((np.unique(addresses),addresses))
    with ExitStack() as resources:
        if config_path:
            from stash_ai.recognition.config import load
            resources.enter_context(scope(load(Path(config_path))))
        db = resources.enter_context(closing(connect()))
        sql_calls = 0
        def statement(_):
            nonlocal sql_calls
            sql_calls+=1
        db.set_trace_callback(statement)
        db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value)')
        index = ReverseIndex(db,'benchmark','library','v1',old)
        tokens = [index.bins(v,.003)|{'o'+digest(str(i)) for i in np.flatnonzero(hit)}
                  for v,hit in zip(vectors,old_hits)]
        t0=time.perf_counter()
        io_start=process_io()
        for i,(unique,_) in enumerate(queries):
            scan=str(i)
            index.post(scan,set().union(*(tokens[k] for k in unique)))
            index.state['scans'][scan]=[scan]
        index.finish()
        db.commit()
        built=time.perf_counter()
        previous_sql=sql_calls
        index=ReverseIndex(db,'benchmark','library','v1',new)
        chosen=index.nominate(new,[])
        def best(values):
            return np.where(values.max(axis=1)>=floor,values.argmax(axis=1),-1)
        expected=np.any(old_hits!=new_hits,axis=1) | (best(old_scores)!=best(new_scores))
        missed=missed_queries=changed_scenes=0
        for i,(unique,addresses) in enumerate(queries):
            changed_scenes+=bool(expected[unique].any())
            if str(i) not in chosen:
                missed+=bool(expected[unique].any())
                missed_queries+=int(expected[addresses].sum())
        finished=time.perf_counter()
        io_end=process_io()
        page_size=db.execute('PRAGMA page_size').fetchone()[0]
        pages=db.execute('PRAGMA page_count').fetchone()[0]
        return dict(scenes=scenes,vtt_cues=scenes*300,positive_cues=scenes*210,body_observations=scenes*420,
                    dimensions=dimensions,distinct_view_vectors=len(vectors),nominated=len(chosen),
                    nomination_fraction=len(chosen)/scenes,exact_changed_scenes=changed_scenes,
                    disagreements=missed,missed_changed_observations=missed_queries,
                    index_build_seconds=round(built-t0,3),nomination_and_verify_seconds=round(finished-built,3),
                    nomination_sql_statements=sql_calls-previous_sql,sqlite_database_bytes=page_size*pages,
                    storage='configured_disk' if config_path else 'memory',
                    process_io_bytes={k:io_end[k]-v for k,v in io_start.items() if k in io_end})
