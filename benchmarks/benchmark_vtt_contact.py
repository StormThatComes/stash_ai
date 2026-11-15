"""Run the face and temporal pose detection paths against an independent WEBVTT sprite."""
import argparse
import json
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from urllib.parse import urlparse

from stash_ai.recognition.config import load
from stash_ai.recognition.detection_audit import audit_scene
from stash_ai.recognition.detection_session import detector_factory
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.models import ensure_assets

DATA = Path(__file__).with_name('data')/'vtt_contact'


class FixtureMedia:
    def get(self,url,**options):
        from requests import Response
        name = Path(urlparse(url).path).name
        if name not in ('contact.vtt','contact.jpg'):
            raise ValueError(f'Unexpected fixture resource {name}')
        response = Response()
        response.status_code = 200
        response.url = url
        response._content = (DATA/name).read_bytes()
        response._content_consumed = True
        return response


def run(config):
    from stash_ai.recognition.detection_masks import MaskOptions
    settings = config.data['models']
    paths = ensure_assets(config,keys=[settings['detector']])
    scene = dict(paths={'vtt':'/contact.vtt','sprite':'/contact.jpg'})
    with TemporaryDirectory() as temporary, SpriteInput(SimpleNamespace(url='http://fixture.invalid',media=FixtureMedia()),scene,config) as src:
        with closing(detector_factory(config,paths)) as detector:
            report = audit_scene(src,detector,'holdout',Path(temporary),mask_options=MaskOptions())
        rows = report['frames']
        if len(rows) != 6 or rows[2]['shot'] == rows[3]['shot']:
            raise AssertionError('The detector did not identify the known edit between cues 2 and 3')
        report['face_score_threshold'] = config.data['detection']['score_threshold']
        for row in rows:
            for person in row['people']:
                source = person.get('pose_refinement',{}).get('temporal_source_cue')
                if source is not None and (source != row['cue_index']-1 or row['cue_index']==3):
                    raise AssertionError('A temporal prompt crossed the known cut or skipped a cue')
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DATA.parents[2]/'facial_recognition.yml')
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--face-score-threshold',type=float,help='Optional diagnostic SCRFD threshold on this holdout')
    args=parser.parse_args()
    config=load(args.config,device=args.device)
    if args.face_score_threshold is not None:
        from .benchmark_dance_contact import apply_edits
        config,_ = apply_edits(config,{'face.detector.score_threshold':args.face_score_threshold})
    if args.model_dir:
        config=replace(config,model_dir=args.model_dir.resolve())
    args.output.write_text(json.dumps(run(config),indent=2)+'\n')


if __name__ == '__main__':
    main()
