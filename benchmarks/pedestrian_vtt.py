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
    labels,missing = {},[]
    with closing(_open_readonly(config.database)) as db:
        _,scans = _selection(db,config,[scene],scope)
        if scans[0]['total']!=len(data['cues']):
            raise ValueError('Expected the prepared eight-cue VTT scan.')
        for cue in data['cues']:
            index = cue['cue']
            rows = []
            for oid,raw,q in db.execute('SELECT id,geometry,quality FROM body_observations '
                    'WHERE scan=? AND cue=? AND contract=? AND rejected IS NULL AND embedding IS NOT NULL',
                    (scans[0]['id'],index,contract(config))):
                if json.loads(q)['quality'].get('frame_signature')!=signatures[index]:
                    raise ValueError(f'Cue {index} does not match the prepared sprite.')
                rows.append((oid,json.loads(raw)['bounds']))
            proposals = []
            for label in cue['labels']:
                ranked = sorted(((overlap(label['box'],box),oid) for oid,box in rows),reverse=True)
                if ranked and ranked[0][0]>=.35 and (len(ranked)==1 or ranked[0][0]-ranked[1][0]>=.12):
                    proposals.append((label,ranked[0][1]))
                else:
                    missing.append([index,label['truth']])
            for label,oid in proposals:
                if sum(other==oid for _,other in proposals)!=1:
                    missing.append([index,label['truth']])
                else:
                    labels[oid] = dict(truth=label['truth'],slices=label['slices'])
    if not labels:
        raise ValueError('No unambiguous measured observations overlap the annotations.')
    manifest = dict(scenes=[scene],database=str(config.database.resolve()),observations=labels,
                    unbound_manual_labels=missing)
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
