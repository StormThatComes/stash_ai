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


def _legacy_load(spool,signed,config):
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


def _legacy_save(spool,signed,modes,support,selected,config):
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


CHUNK=1024*1024


def load(spool,signed,config):
    store=current()
    record=store.get('body-appearances',signed) if store else None
    if record is None:return None
    if record.get('format')!=3:return _legacy_load(spool,signed,config)
    written=[];modes=[];complete=False
    try:
        for index,(count,values,weights) in enumerate(record['modes']):
            if type(count) is not int or count<0:return None
            paths=[]
            for suffix,parts,expected in (('.bin',values,count*16384),('.weights',weights,count)):
                path=spool.path/(str(index)+suffix);written.append(path);paths.append(path);total=0
                with path.open('wb') as output:
                    for key,length in parts:
                        if not 0<=length<=CHUNK:return None
                        data=store.get_bytes('body-appearance-chunk',key)
                        if data is None or len(data)!=length:return None
                        total+=len(data)
                        if total>expected:return None
                        output.write(data)
                if total!=expected:return None
            modes.append(Matrix(paths[0],count,spool.directory,paths[1]))
        complete=True
        return tuple(modes),record['support'],record['selected']
    except (ValueError,KeyError,TypeError):return None
    finally:
        if not complete:
            for path in written:path.unlink(missing_ok=True)


def save(spool,signed,modes,support,selected,config):
    store=current()
    if store is None:return
    if not hasattr(store,'put_bytes'):return _legacy_save(spool,signed,modes,support,selected,config)
    capacity=store.capacity('body-appearances') if hasattr(store,'capacity') else store.budget
    if sum(len(mode)*16385 for mode in modes)>capacity//2:return
    values=[]
    for mode in modes:
        packed=[]
        for path in (mode.path,mode.weights_path):
            parts=[]
            with open(path,'rb') as source:
                while chunk:=source.read(CHUNK):
                    key=hashlib.sha256(chunk).hexdigest()
                    store.put_bytes('body-appearance-chunk',key,chunk);parts.append([key,len(chunk)])
            packed.append(parts)
        values.append([len(mode),*packed])
    witnesses=[row[0] for row in spool.db.execute('SELECT id FROM records ORDER BY ordinal')]
    # Publish the manifest only after all chunks. A missing chunk is a miss.
    store.put('body-appearances',signed,dict(format=3,modes=values,support=support,selected=selected,witnesses=witnesses))


def witnesses(signed):
    store=current();record=store.get('body-appearances',signed) if store else None
    return record.get('witnesses') if record else None
