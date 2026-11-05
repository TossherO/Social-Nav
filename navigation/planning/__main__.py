"""Explicit build, geometric query, replay and read-only review entry point."""
import argparse
import json
from pathlib import Path
from .backend import build
from .footprint import VehicleGeometry
from .inputs import load_config, load_input
from .planner import Planner
from .replay import replay, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    b = commands.add_parser('build'); b.add_argument('--ros-prefix')
    cuda=commands.add_parser('build-cuda',help='Build the standalone CUDA sweep library for an explicit device architecture')
    cuda.add_argument('--arch',required=True)
    people=commands.add_parser('inspect-people',help='Read-only diagnostic review of recorded shadow people checks')
    people.add_argument('--run',required=True);people.add_argument('--output',required=True)
    people.add_argument('--samples',type=int,default=5);people.add_argument('--no-assets',action='store_true')
    people.add_argument('--compare-config',help='Explicit planning YAML for a same-record people-model comparison')
    plan = commands.add_parser('plan')
    plan.add_argument('--mode', required=True, choices=['static', 'mapped', 'local'])
    plan.add_argument('--static-map'); plan.add_argument('--scene')
    plan.add_argument('--frame')
    plan.add_argument('--endpoints', help='Explicit static points produced by pick')
    plan.add_argument('--start', type=float, nargs=3, metavar=('X', 'Y', 'YAW'))
    plan.add_argument('--goal', type=float, nargs='+', help='X Y [YAW], in metres/radians')
    plan.add_argument('--algorithm', choices=['dijkstra', 'astar','hybrid_astar'])
    plan.add_argument('--task-id', default='manual_query'); plan.add_argument('--goal-revision', type=int, default=0)
    plan.add_argument('--request-id', default='manual_query')
    run = commands.add_parser('replay'); run.add_argument('--manifest', required=True)
    online=commands.add_parser('online',help='ROS planning-only grid/pose/goal input and timed Path output')
    online.add_argument('--geometry',choices=['historical'],required=True)
    online.add_argument('--config');online.add_argument('--motion-config');online.add_argument('--runtime-config')
    online.add_argument('--environment-config');online.add_argument('--static-map')
    online.add_argument('--grid-topic',default='/perception/environment/grid')
    online.add_argument('--state-topic',default='/localization/state')
    online.add_argument('--people-topic',default='/perception/people/json')
    online.add_argument('--shadow',action='store_true',help='Publish private tracking proposals for the independent shadow gate; never chassis commands')
    online.add_argument('--shadow-config')
    online.add_argument('--collision-backend',choices=['cpu','cuda'])
    online.add_argument('--dwa-config')
    route=commands.add_parser('route',help='Plan one explicitly selected mixed-route leg')
    route.add_argument('--route',required=True);route.add_argument('--segment',required=True)
    route.add_argument('--scene');route.add_argument('--static-map');route.add_argument('--anchor-scene')
    route.add_argument('--progress-m',type=float,default=0.);route.add_argument('--lookahead-m',type=float,default=2.)
    route.add_argument('--task-id',default='route_query');route.add_argument('--goal-revision',type=int,default=0)
    for command in (plan, run, route):
        command.add_argument('--config')
        command.add_argument('--geometry', choices=['historical'], required=True,
                             help='Explicit assumption for offline diagnostics; never executable')
        command.add_argument('--output', required=True)
    view = commands.add_parser('view'); view.add_argument('--results', required=True)
    view.add_argument('--rviz', action='store_true', help='Publish only private navigation display topics; needs a ROS master')
    pick = commands.add_parser('pick'); pick.add_argument('--static-map', required=True)
    pick.add_argument('--output', required=True, help='New JSON file containing explicit static start/goal')
    args = parser.parse_args(argv)
    if args.command=='build-cuda':
        from .collision import build_cuda
        print(build_cuda(args.arch));return
    if args.command=='inspect-people':
        from .people_review import inspect_people
        print(json.dumps(inspect_people(args.run,args.output,args.samples,assets=not args.no_assets,compare_config=args.compare_config),ensure_ascii=False));return
    if args.command == 'build': build(args.ros_prefix); return
    if args.command == 'pick':
        from .visualization import pick_endpoints
        pick_endpoints(args.static_map, args.output); return
    if args.command == 'view':
        from .visualization import review, rviz_view
        if args.rviz: rviz_view(args.results)
        else: print(review(args.results))
        return
    config = load_config(args.config)
    geometry = VehicleGeometry.historical(config['geometry_reference'])
    if args.command=='online':
        from .online import run
        from .motion import load_motion_config
        motion=load_motion_config(args.motion_config)
        if args.collision_backend:motion['collision_backend']=args.collision_backend
        from .reference import load_config as reference_config
        settings=reference_config(args.runtime_config)
        run(args,config,motion,geometry,settings);return
    planner = Planner(config, geometry)
    if args.command == 'replay':
        rows = replay(args.manifest, args.output, planner)
    elif args.command=='route':
        from .route import RoutePlan
        reference=RoutePlan(json.loads(Path(args.route).read_text()))
        segment=reference.segments[args.segment]
        if segment['mode']!='transition' and not args.scene:parser.error('Route geometry requires --scene')
        if args.anchor_scene and segment['mode']!='local':parser.error('--anchor-scene is for local route legs')
        scene,asset=load_input(scene=args.scene,static_map=args.static_map) if args.scene or args.static_map else (None,None)
        anchor=None
        if args.anchor_scene:
            anchor_scene=load_input(scene=args.anchor_scene)[0]
            anchor=reference.bind_anchor(segment['anchor_id'],anchor_scene,now=anchor_scene.record['as_of'],
                clock_domain=anchor_scene.record['clock_domain'],config=config)
        selection=reference.request(args.segment,scene,geometry,task_id=args.task_id,goal_revision=args.goal_revision,
            now=None if scene is None else scene.record['as_of'],clock_domain=None if scene is None else scene.record['clock_domain'],
            config=config,progress_m=args.progress_m,lookahead_m=args.lookahead_m,asset=asset,anchor=anchor)
        target=Path(args.output);target.mkdir(parents=True,exist_ok=False)
        write_json(target/'route_reference.json',selection)
        if selection['request'] is None:
            rows=[dict(label=args.segment,event=selection['status'],status='unavailable',executable=False)]
        else:
            result=planner.query(selection['request'],scene=scene,asset=asset,now=scene.record['as_of'],clock_domain=scene.record['clock_domain'])
            rows=[dict(label=args.segment,result=result,source=dict(scene=args.scene,static_map=args.static_map,
                request=selection['request']),evaluation_mode='explicit_route_reference')]
        write_json(target/'results.json',dict(queries=rows))
    else:
        scene, asset = load_input(scene=args.scene, static_map=args.static_map)
        if args.endpoints:
            if args.mode != 'static' or args.start is not None or args.goal is not None or args.frame is not None:
                parser.error('--endpoints is for static mode and cannot be mixed with explicit start/goal/frame')
            points = json.loads(Path(args.endpoints).read_text())
            if points.get('target_reference') != 'base_link': parser.error('Selected endpoints must reference base_link')
            if asset is None or points['map_snapshot_id'] != asset.manifest['snapshot_id']:
                parser.error('Selected endpoints belong to a different map')
            args.start, args.goal, args.frame = points['start'], points['goal'], points['frame_id']
        if args.goal is None or args.frame is None: parser.error('Provide --goal and --frame, or --endpoints')
        request = dict(request_id=args.request_id, task_id=args.task_id, goal_revision=args.goal_revision,
                       mode=args.mode, frame_id=args.frame, target_reference='base_link', goal=args.goal)
        if args.start is not None: request['start'] = args.start
        if args.algorithm: request['algorithm'] = args.algorithm
        if asset: request['map_snapshot_id'] = asset.manifest['snapshot_id']
        result = planner.query(request, scene=scene, asset=asset,
            now=None if scene is None else scene.record['as_of'],
            clock_domain=None if scene is None else scene.record['clock_domain'])
        source = dict(scene=args.scene, static_map=args.static_map, request=request)
        rows = [dict(label=args.request_id, result=result, source=source, evaluation_mode='recorded_observation_time')]
        target = Path(args.output); target.mkdir(parents=True, exist_ok=False)
        write_json(target/'results.json', dict(queries=rows))
    print(json.dumps([dict(label=r['label'], status=r.get('result', r)['status'],
        reason=r.get('result', r).get('reason')) for r in rows], ensure_ascii=False))


if __name__ == '__main__': main()
