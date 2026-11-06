"""Bounded, checksummed appearance banks; no persisted Python object loading."""
import base64
import hashlib
import zlib
from .config import digest
from .derived_cache import current
from .disk_body_modes import Matrix


def key(spool,config):
    h=hashlib.sha256()
    for (payload,) in spool.db.execute('SELECT payload FROM records ORDER BY ordinal'):h.update(payload)
    return digest(['body-appearance-v2',h.hexdigest(),config.data['sapiensid']['matching'],
                   config.data['clustering']['max_samples_per_person']])


def load(spool,signed,config):
    store=current()
    record=store.get('body-appearances',signed) if store else None
    if record is None:return None
    maximum=min(store.budget//32,config.data['sapiensid']['execution']['reference_bytes']//8)
    expected=sum(n*(4096*4+1) for n,_,_ in record['modes'])
    if expected>maximum:return None
    modes=[];written=[];complete=False
    try:
        for i,(count,values,weights) in enumerate(record['modes']):
            files=[]
            for suffix,payload,length in (('.bin',values,count*4096*4),('.weights',weights,count)):
                decoder=zlib.decompressobj()
                raw=decoder.decompress(base64.b64decode(payload,validate=True),length+1)
                if len(raw)!=length or not decoder.eof or decoder.unused_data:return None
                path=spool.path/(str(i)+suffix);path.write_bytes(raw);files.append(path);written.append(path)
            modes.append(Matrix(files[0],count,spool.directory,files[1]))
        complete=True
        return tuple(modes),record['support'],record['selected']
    except (ValueError,zlib.error,TypeError,KeyError):return None
    finally:
        if not complete:
            for path in written:path.unlink(missing_ok=True)


def save(spool,signed,modes,support,selected,config):
    store=current()
    if store is None:return
    maximum=min(store.budget//32,config.data['sapiensid']['execution']['reference_bytes']//8)
    if sum(len(m)*(4096*4+1) for m in modes)>maximum:return
    values=[]
    for mode in modes:
        def pack(path):return base64.b64encode(zlib.compress(path.read_bytes(),1)).decode('ascii')
        from pathlib import Path
        values.append([len(mode),pack(Path(mode.path)),pack(Path(mode.weights_path))])
    store.put('body-appearances',signed,dict(modes=values,support=support,selected=selected))
