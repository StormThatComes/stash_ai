"""Prepare the public pedestrian VTT sample or bind its manual labels to a retained scan."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image
from stash_ai.recognition.body_perspectives import frame_signature


LABELS = Path(__file__).with_name('data')/'pedestrian_vtt.json'


def prepare(video, output, data):
    with video.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=data['sha256']:
            raise ValueError('Source video checksum differs from the labelled sample.')
    output.mkdir(parents=True,exist_ok=True)
    w,h = data['size']
    cues = data['cues']
    select = '+'.join(f"eq(n,{cue['frame']})" for cue in cues)
    raw = subprocess.run(['ffmpeg','-v','error','-i',str(video),'-vf',f"select='{select}',scale={w}:{h}",
                          '-vsync','0','-f','rawvideo','-pix_fmt','rgb24','-'],check=True,stdout=subprocess.PIPE).stdout
    stride = w*h*3
    if len(raw)!=len(cues)*stride:
        raise ValueError('Frame extraction did not produce all annotated cues.')
    signatures = []
    with Image.new('RGB',(w*len(cues),h)) as sprite:
        vtt = ['WEBVTT','']
        for i,cue in enumerate(cues):
            with Image.frombytes('RGB',(w,h),raw[i*stride:(i+1)*stride]) as frame:
                signatures.append(frame_signature(frame))
                sprite.paste(frame,(i*w,0))
            sec = cue['frame']/data['fps']
            vtt.extend([f'00:00:{sec:06.3f} --> 00:00:{sec+.5:06.3f}',
                        f'pedestrians.png#xywh={i*w},0,{w},{h}',''])
        sprite.save(output/'pedestrians.png')
    (output/'pedestrians.vtt').write_text('\n'.join(vtt)+'\n')
    (output/'signatures.json').write_text(json.dumps(signatures))


def overlap(a,b):
    inter = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def bind(config, scope, scene, signatures, output, data):
    from stash_ai.recognition.body_observations import contract
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.offline_evaluation import _selection
    labels,missing,visible = {},[],{}
    with closing(_open_readonly(config.database)) as db:
        _,scans = _selection(db,config,[scene],scope)
        if scans[0]['total']!=len(data['cues']):
            raise ValueError('Expected the prepared eight-cue VTT scan.')
        for cue in data['cues']:
            index = cue['cue']
            rows = []
            visible[str(index)] = []
            for oid,raw,q,ready in db.execute('SELECT id,geometry,quality,(rejected IS NULL AND embedding IS NOT NULL) FROM body_observations '
                    'WHERE scan=? AND cue=? AND contract=?',
                    (scans[0]['id'],index,contract(config))):
                if ready and json.loads(q)['quality'].get('frame_signature')!=signatures[index]:
                    raise ValueError(f'Cue {index} does not match the prepared sprite.')
                bounds = json.loads(raw).get('bounds')
                if bounds:
                    rows.append((oid,bounds,ready))
            proposals = []
            for label in cue['labels']:
                x,y,r,b = label['box']
                person = dict(truth=label['truth'], point=[(x+r)/2,(y+b)/2], slices=label['slices'])
                visible[str(index)].append(person)
                ranked = sorted(((overlap(label['box'],box),oid,ready) for oid,box,ready in rows),reverse=True)
                if not ranked or ranked[0][0]<.35:
                    person['binding'] = 'missing'
                elif len(ranked)>1 and ranked[0][0]-ranked[1][0]<.12:
                    person['binding'] = 'ambiguous'
                elif not ranked[0][2]:
                    person['binding'] = 'unavailable'
                else:
                    proposals.append((person,ranked[0][1]))
            for person,oid in proposals:
                if sum(peer==oid for _,peer in proposals)!=1:
                    person['binding'] = 'ambiguous'
                else:
                    person['binding'] = 'bound'
                    labels[oid] = dict(truth=person['truth'],slices=person['slices'])
            missing.extend([index,p['truth']] for p in visible[str(index)] if p['binding']!='bound')
    manifest = dict(scenes=[scene],database=str(config.database.resolve()),observations=labels,
                    visible_people={scene:visible},unbound_manual_labels=missing)
    with output.open('x') as stream:
        json.dump(manifest,stream,indent=2)
    print(f'{len(labels)} bound, {len(missing)} missing or ambiguous')


def main():
    from stash_ai.recognition.config import load
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode',required=True)
    make = modes.add_parser('prepare')
    make.add_argument('video',type=Path)
    make.add_argument('directory',type=Path)
    link = modes.add_parser('bind')
    for name in ('config','signatures','output'):
        link.add_argument('--'+name,required=True,type=Path)
    link.add_argument('--scene',required=True)
    link.add_argument('--scope',required=True)
    args = parser.parse_args()
    data = json.loads(LABELS.read_text())
    if args.mode=='prepare':
        prepare(args.video,args.directory,data)
    else:
        bind(load(args.config),args.scope,args.scene,json.loads(args.signatures.read_text()),args.output,data)


if __name__=='__main__':
    main()
