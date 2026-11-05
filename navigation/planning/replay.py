"""Replay complete saved observations; never reconstruct grids from metadata."""
import json
from pathlib import Path
from .backend import ROOT
from .inputs import load_input, scene_status, association_status
from .planner import PlanningSession


def resolve(path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT/p


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def replay(manifest, output, planner):
    specification = json.loads(Path(manifest).read_text())
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    session = PlanningSession(); rows = []; cache = {}
    for index, entry in enumerate(specification['queries']):
        if entry.get('event') in ('seek', 'cancel'):
            session.invalidate(entry['event'])
            if entry['event']=='seek':planner.reset()
            rows.append(dict(label=entry.get('label', entry['event']), event=entry['event'],
                             status='unavailable', reason=entry['event'], executable=False))
            continue
        scene_path = resolve(entry['scene']) if entry.get('scene') else None
        map_path = resolve(entry['static_map']) if entry.get('static_map') else None
        scene = asset = None
        if scene_path:
            key = ('scene', str(scene_path))
            if key not in cache: cache[key] = load_input(scene=scene_path)[0]
            scene = cache[key]
            # Continuous evidence replay must not retain every decompressed grid.
            scenes=[k for k in cache if k[0]=='scene']
            if len(scenes)>8:del cache[scenes[0]]
        if map_path:
            key = ('map', str(map_path))
            if key not in cache: cache[key] = load_input(static_map=map_path)[1]
            asset = cache[key]
        if entry.get('event')=='observe':
            session.invalidate('new_observation')
            try:
                import numpy as np
                if scene is None or asset is None:raise ValueError('Observe needs a scene and static map')
                reason=scene_status(scene,scene.record['as_of'],scene.record['clock_domain'],planner.config) or association_status(scene,asset,planner.config)
                if reason:raise ValueError(reason)
                pose=np.asarray(scene.record['prior']['map_from_local']) @ np.asarray(scene.record['localization']['world_from_lidar'])
                bounds=np.r_[pose[:2,3]-15.,pose[:2,3]+15.]
                revised=planner.observe(scene,asset,now=scene.record['as_of'],
                    clock_domain=scene.record['clock_domain'],bounds=bounds)
                rows.append(dict(label=entry.get('label',str(index)),event='observe',status='observed',
                    prior_revision=revised.metadata['prior_revision']))
            except (ValueError,KeyError,TypeError) as error:
                rows.append(dict(label=entry.get('label',str(index)),event='observe',status='unavailable',reason=str(error)))
            continue
        request = entry['request']
        try:
            ticket = session.begin(request, scene, asset)
            result = planner.query(request, scene=scene, asset=asset,
                now=None if scene is None else scene.record['as_of'],
                clock_domain=None if scene is None else scene.record['clock_domain'])
            accepted = session.accept(ticket, result)
        except ValueError as error:
            result = dict(status='invalid_input', reason=str(error), executable=False, path=[], request=request)
            accepted = False
        row = dict(label=entry.get('label', str(index)), source=entry, result=result,
                   accepted=accepted, evaluation_mode='recorded_observation_time')
        rows.append(row)
    write_json(target/'results.json', dict(source_manifest=str(Path(manifest).resolve()), queries=rows))
    return rows
