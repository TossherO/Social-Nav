"""Prepare and run T3 seating inputs; ROS extraction uses the system interpreter."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    extract = commands.add_parser('extract', help='Read causal bag observations with the system ROS interpreter')
    extract.add_argument('--run', type=Path, required=True)
    extract.add_argument('--bag', type=Path, required=True)
    extract.add_argument('--offsets', nargs='+', type=float, required=True)
    extract.add_argument('--out', type=Path, required=True)
    extract.add_argument('--clock-domain', help='Only required for an unspecified legacy source; must match recorded domains')
    extract.add_argument('--scenes', type=Path, nargs='+', help='Same-time current scene, or one per offset, for navigation binding')

    window = commands.add_parser('prepare-window', help='Three causal RGB+BEV panels at -1/-0.5/0 seconds')
    window.add_argument('--observations', type=Path, nargs=3, required=True)
    source = window.add_mutually_exclusive_group(required=True)
    source.add_argument('--candidates', type=Path)
    source.add_argument('--table-prior', type=Path)
    window.add_argument('--vehicle-reference', type=Path, help='Explicit historical geometry asset for prior sampling')
    window.add_argument('--instruction', help='Seating instruction for --table-prior; existing candidate instructions stay frozen')
    window.add_argument('--out', type=Path, required=True)
    window.add_argument('--current-only', action='store_true', help='Explicit single-frame ablation')

    run = commands.add_parser('run', help='Call remote DSPARK using frozen input')
    run.add_argument('--input', type=Path, required=True)
    run.add_argument('--base-url', required=True)
    run.add_argument('--prompt', type=Path)
    run.add_argument('--result-name', default='remote')

    check = commands.add_parser('check-navigation', help='Check frozen candidate poses against a same-observation navigation scene')
    check.add_argument('--input', type=Path, required=True)
    check.add_argument('--scene', type=Path, required=True)
    check.add_argument('--out', type=Path, required=True)

    single = commands.add_parser('prepare', help='Prepare a single SceneInputs or raw-scan diagnostic')
    source = single.add_mutually_exclusive_group(required=True)
    source.add_argument('--scene', type=Path)
    source.add_argument('--observation', type=Path)
    single.add_argument('--candidates', type=Path, required=True)
    single.add_argument('--out', type=Path, required=True)

    edge = commands.add_parser('prepare-edge', help='Fit assisted causal edges from raw observations')
    edge.add_argument('--observations', type=Path, required=True)
    edge.add_argument('--rois', type=Path, required=True, help='Ordered name -> current-image XYXY or null')
    edge.add_argument('--camera', type=Path, required=True, help='CameraInfo K/D/width/height JSON')
    edge.add_argument('--reference', type=Path, required=True)
    edge.add_argument('--calibration', type=Path, required=True)
    edge.add_argument('--reviewed-edge', type=Path)
    edge.add_argument('--front-gaps', type=float, nargs='+', required=True)
    edge.add_argument('--end-guard', type=float, required=True)
    edge.add_argument('--out', type=Path, required=True)

    refine = commands.add_parser('refine-edge', help='Refine frozen causal edges with an explicit seating policy')
    refine.add_argument('--previous-review', type=Path, required=True)
    refine.add_argument('--policy', type=Path, required=True)
    refine.add_argument('--out', type=Path, required=True)

    args = parser.parse_args(argv)
    # Import only the selected operation: extract must not load Conda-only code.
    if args.command == 'extract':
        from .inputs import extract
        extract(args.run, args.bag, args.out, args.offsets,clock_domain=args.clock_domain,scenes=args.scenes)
    elif args.command == 'prepare-window':
        from .inputs import freeze_window
        data = args.candidates
        if args.candidates and (args.instruction is not None or args.vehicle_reference is not None):
            parser.error('--instruction and --vehicle-reference apply only to --table-prior')
        if args.table_prior:
            if args.vehicle_reference is None:
                parser.error('--table-prior requires explicit --vehicle-reference')
            instruction = args.instruction if args.instruction is not None else '在已标注的这张桌子旁选择方便的轮椅就座位置，允许略斜停靠，脚踏可以进入桌下。'
            if not instruction.strip():
                parser.error('--instruction must not be empty')
            from ..project_config import load_config, project_path
            from ..vlm_api_common import read_json
            from .geometry import generate, read_observation, IDENTITY
            task = load_config()['tasks']['seating']
            policy = read_json(project_path(task['geometry_policy']))
            context, goals, _ = generate(read_json(args.table_prior), args.observations, args.vehicle_reference, policy)
            current = read_observation(args.observations[-1])
            data = {k: current[k] for k in (*IDENTITY, 'observation_id', 'stamp_s')}
            data.update(instruction=instruction, revision=0, context=context, candidates=goals)
        freeze_window(args.observations, data, args.out, current_only=args.current_only)
    elif args.command == 'check-navigation':
        from perception.environment.scene import SceneInputs
        from navigation.planning.inputs import load_config
        from navigation.planning.motion import load_motion_config
        from navigation.planning.footprint import VehicleGeometry
        from ..annotation.rules import digest
        from .geometry import check_navigation
        item=json.loads((args.input/'input.json').read_text())
        if digest({k:v for k,v in item.items() if k!='input_fingerprint'})!=item['input_fingerprint']:
            raise ValueError('Frozen input binding changed')
        scene=SceneInputs.load(args.scene);cfg=load_config()
        if any(args.out.resolve().is_relative_to(p.resolve()) for p in (args.input,args.scene)):
            raise ValueError('Output must be outside frozen inputs')
        if args.out.exists():raise FileExistsError(args.out)
        result=check_navigation(item['candidate_set'],scene,cfg,load_motion_config(),
            VehicleGeometry.historical(cfg['geometry_reference']),now=scene.record['as_of'])
        args.out.mkdir(parents=True)
        (args.out/'navigation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    elif args.command == 'run':
        from .inference import run
        print(json.dumps(run(args.input, args.base_url, prompt_path=args.prompt, result_name=args.result_name), ensure_ascii=False))
    elif args.command == 'prepare':
        from .inputs import freeze_scene, freeze_scan
        (freeze_scene if args.scene else freeze_scan)(args.scene or args.observation, args.candidates, args.out)
    else:
        from .inputs import prepare_review, refine_review
        if args.command == 'refine-edge':
            result = refine_review(args.previous_review, args.policy, args.out)
        else:
            result = prepare_review(args.observations, json.loads(args.rois.read_text()), args.reference,
                args.calibration, json.loads(args.camera.read_text()), args.out,
                reviewed_edge_path=args.reviewed_edge, front_gaps_m=args.front_gaps, end_guard_m=args.end_guard)
        print(json.dumps([dict(case=c['case'], status=c['status'], candidates=len(c['candidate_set']['candidates']))
                          for c in result['cases']]))


if __name__ == '__main__':
    main()
