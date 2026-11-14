"""Inspect, export and query geometric environment maps."""
import argparse
import json
import math
from pathlib import Path
import sys

from .layers import GridSnapshot
from .context import query_geometry


def bundle_main(argv):
    from .export import export_environment, export_endpoint_environment, export_registered_environment
    from .storage import StaticEnvironmentMap
    from .layers import load_config, STATIC_FILTER_MODES
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    rebuild = commands.add_parser('rebuild-scenes',help='Causally rebuild local geometry from raw scans and saved poses')
    rebuild.add_argument('--run',required=True);rebuild.add_argument('--scenes',nargs='+',required=True)
    rebuild.add_argument('--out',required=True);rebuild.add_argument('--config')
    rebuild.add_argument('--environment-backend',choices=['cpu','cuda'])
    window=commands.add_parser('replay-window',help='Single-pass causal geometry replay at saved scene cutoffs')
    window.add_argument('--run',required=True);window.add_argument('--out',required=True)
    window.add_argument('--start-s',type=float,required=True);window.add_argument('--end-s',type=float,required=True)
    window.add_argument('--warmup-s',type=float)
    window.add_argument('--sampling',choices=['all','recorded'],default='all')
    window.add_argument('--config')
    window.add_argument('--environment-backend',choices=['cpu','cuda'])
    people = commands.add_parser('prepare-people', help='Infer source-bound people for a recorded mapping/localization run')
    people.add_argument('--run', required=True)
    people.add_argument('--out', required=True)
    people.add_argument('--detector-python', required=True)
    people.add_argument('--gpu', required=True, help='Explicit physical GPU index for the detector worker')
    people.add_argument('--detector-config')
    export = commands.add_parser('export', help='Build a geometric environment package from a recorded mapping run')
    export_source = export.add_mutually_exclusive_group(required=True)
    export_source.add_argument('--run')
    export_source.add_argument('--localization-map', help='Target ROLL map: registered scan evidence, or explicit endpoint-only --dynamic-filter none')
    export.add_argument('--evidence-run',action='append',help='Sealed scan run; repeat with paired --reference-run for multiple surveys')
    export.add_argument('--reference-run',action='append',help='Same-bag localization on target map; repeat in evidence-run order')
    export.add_argument('--out', required=True)
    export.add_argument('--config')
    export.add_argument('--resolution', type=float, help='Static output grid and fusion voxel resolution in metres')
    export.add_argument('--bounds',type=float,nargs=4,metavar=('XMIN','YMIN','XMAX','YMAX'),
                        help='Explicit source-map region, aligned outward to cells; omitted keeps the whole map')
    export.add_argument('--bag', help='Explicit relocation of the same bound source bag')
    export.add_argument('--people', help='Source-bound environment_people package from this mapping run')
    export.add_argument('--dynamic-filter', choices=STATIC_FILTER_MODES,
                        help='Default: static.dynamic_filter (geometric); sealed scans required, people optional; none allows endpoint-only export')
    export.add_argument('--projection-strategy',choices=['observed_endpoints','ground_relative_assume_free','docker_assume_free'],
                        default='observed_endpoints',help='Endpoint projection; assumed-free strategies do not provide ray evidence')
    export.add_argument('--projection-z-min',type=float,help='Legacy absolute map Z minimum')
    export.add_argument('--projection-z-max',type=float,help='Legacy absolute map Z maximum')
    export.add_argument('--projection-height-min',type=float,help='Minimum height above local ground in metres')
    export.add_argument('--projection-height-max',type=float,help='Maximum height above local ground in metres')
    export.add_argument('--projection-radius',type=float)
    export.add_argument('--projection-min-neighbors',type=int)
    export.add_argument('--export-pgm','--navigation',dest='export_pgm',action='store_true',
                        help='Export the occupancy layer as map_server PGM/YAML; does not build a navigation costmap')
    query = commands.add_parser('query', help='Read geometry; never invokes models or robot motion')
    query.add_argument('--static-map','--map',dest='map',required=True,help='Static environment map package, not a localization map')
    query.add_argument('--operation', choices=['summary','geometry','crop'], default='summary')
    query.add_argument('--frame')
    query.add_argument('--position', type=float, nargs=2)
    query.add_argument('--bounds', type=float, nargs=4)
    query.add_argument('--layers', nargs='+', default=['occupancy'])
    view = commands.add_parser('view', help='View the geometric map layers')
    view.add_argument('--static-map','--map',dest='map',required=True,help='Static environment map package, not a localization map')
    view.add_argument('--out', help='New overview image')
    view.add_argument('--show', action='store_true')
    args = parser.parse_args(argv)
    if args.command in ('replay-window','rebuild-scenes'):
        replay_config=load_config(args.config)
        if args.environment_backend:
            replay_config['compute']['backend']=args.environment_backend
    if args.command=='replay-window':
        from .replay import replay_window
        result=replay_window(args.run,args.out,start_s=args.start_s,end_s=args.end_s,config=replay_config,
                             warmup_s=args.warmup_s,sampling=args.sampling)
        print(json.dumps(dict(out=args.out,frames=len(result['frames']))));return
    if args.command == 'rebuild-scenes':
        from .replay import rebuild_scenes
        rebuild_scenes(args.run,args.scenes,args.out,config=replay_config);return
    if args.command == 'prepare-people':
        from .people import prepare_people
        result = prepare_people(args.run, args.out, detector_python=args.detector_python,
                                gpu=args.gpu, detector_config=args.detector_config)
        print(json.dumps(dict(out=args.out, scans=result['scans'], valid=result['valid'])))
        return
    if args.command == 'export':
        config = load_config(args.config)
        def progress(message):
            print(message, file=sys.stderr, flush=True)
        if args.resolution is not None:
            if not math.isfinite(args.resolution) or args.resolution <= 0:
                parser.error('--resolution must be finite and positive')
            config['resolution'] = args.resolution
        projection_options=(args.projection_z_min,args.projection_z_max,args.projection_radius,args.projection_min_neighbors,
                            args.projection_height_min,args.projection_height_max)
        if args.projection_strategy=='observed_endpoints' and any(value is not None for value in projection_options):
            parser.error('Point projection filter options require an assumed-free strategy')
        if args.projection_strategy!='observed_endpoints' and (not args.localization_map or args.evidence_run or args.reference_run):
            parser.error('Assumed-free projection requires endpoint-only --localization-map')
        if args.projection_strategy=='ground_relative_assume_free' and (args.projection_z_min is not None or args.projection_z_max is not None):
            parser.error('Ground-relative projection uses --projection-height-min/max, not absolute Z')
        if args.projection_strategy=='docker_assume_free' and (args.projection_height_min is not None or args.projection_height_max is not None):
            parser.error('Legacy projection uses --projection-z-min/max, not relative heights')
        if args.localization_map:
            if args.evidence_run or args.reference_run:
                if not (args.evidence_run and args.reference_run) or args.bag or args.people or args.dynamic_filter not in (None,'geometric'):
                    parser.error('Registered ray export requires both runs and geometric filtering')
                if len(args.evidence_run) != len(args.reference_run):
                    parser.error('Each --evidence-run needs one matching --reference-run in the same order')
                if len(args.evidence_run) == 1:
                    data=export_registered_environment(args.localization_map,args.evidence_run[0],args.reference_run[0],args.out,
                        config=config,bounds=args.bounds,export_pgm=args.export_pgm,progress=progress)
                else:
                    from .multi_export import export_registered_multi_environment
                    data=export_registered_multi_environment(args.localization_map,args.evidence_run,args.reference_run,args.out,
                        config=config,bounds=args.bounds,export_pgm=args.export_pgm,progress=progress)
            elif args.dynamic_filter != 'none' or args.bag or args.people:
                parser.error('--localization-map requires --dynamic-filter none and cannot accept --bag or --people')
            else:
                settings=({key:value for key,value in zip(('z_min','z_max','radius','min_neighbors','height_min','height_max'),projection_options)
                           if value is not None} if args.projection_strategy!='observed_endpoints' else None)
                data = export_endpoint_environment(args.localization_map,args.out,config=config,
                    bounds=args.bounds,export_pgm=args.export_pgm,projection_strategy=args.projection_strategy,
                    projection_settings=settings)
        else:
            if args.evidence_run or args.reference_run:parser.error('Registered rays require --localization-map')
            data = export_environment(args.run,args.out,config=config,bag=args.bag,
                                      export_pgm=args.export_pgm,people=args.people,filter_mode=args.dynamic_filter,
                                      bounds=args.bounds,progress=progress)
        result=dict(out=args.out,entity_type='map',map_type='static_environment_map',snapshot_id=data['snapshot_id'])
        if args.projection_strategy in ('docker_assume_free','ground_relative_assume_free'):
            result.update(projection_strategy=args.projection_strategy,free_assumption='no_filtered_endpoint_in_cell',
                          free_space_supported=False)
            print('WARNING: non-hit cells are assumed free, not observed; PGM does not preserve this distinction.',file=sys.stderr)
        print(json.dumps(result,ensure_ascii=False))
        return
    package = StaticEnvironmentMap(args.map)
    try:
        if args.command == 'view':
            if not args.out and not args.show:
                parser.error('Choose --out or --show')
            if args.out and Path(args.out).exists():
                raise FileExistsError('Select a new overview image path')
            if not args.show:
                import matplotlib
                matplotlib.use('Agg')
            from perception.visualization.environment import plot_environment
            figure = plot_environment(package.snapshot)
            if args.out:
                figure.savefig(args.out,dpi=130)
            if args.show:
                import matplotlib.pyplot as plt
                plt.show()
            result = dict(out=args.out,snapshot_id=package.manifest['snapshot_id'])
        elif args.operation == 'summary':
            result = package.summary()
        elif args.operation == 'geometry':
            if args.position is None:
                parser.error('Geometry queries require --position X Y')
            result = package.query_geometry([args.position],frame=args.frame)
        else:
            if args.bounds is None:
                parser.error('Map crops require --bounds xmin ymin xmax ymax')
            result = package.crop_map(args.bounds,frame=args.frame,layers=args.layers)
        from .context import json_safe
        print(json.dumps(json_safe(result),ensure_ascii=False,allow_nan=False))
    finally:
        package.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ('export', 'query', 'view', 'prepare-people','rebuild-scenes','replay-window'):
        return bundle_main(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.epilog = 'Package commands: export, query, view, prepare-people, rebuild-scenes, replay-window (COMMAND --help).'
    parser.add_argument('--snapshot',required=True,help='Geometry grid snapshot: local map, static map or derived view; inspect metadata for its role')
    parser.add_argument('--query',type=float,nargs=2,metavar=('X','Y'))
    parser.add_argument('--image',help='New overview image; no overwrite')
    parser.add_argument('--show',action='store_true')
    args = parser.parse_args(argv)
    snapshot = GridSnapshot.load(args.snapshot)
    row = dict(snapshot.metadata,cell_counts={str(k):int((snapshot.layers['occupancy']==k).sum()) for k in [-1,0,100]})
    if args.query is not None:
        row['query'] = query_geometry(snapshot,[args.query],frame=snapshot.metadata['frame_id'])
    print(json.dumps(row,ensure_ascii=False,indent=2,allow_nan=False))
    if args.image or args.show:
        if args.image and Path(args.image).exists():
            parser.error('Refusing to overwrite an existing image')
        if not args.show:
            import matplotlib
            matplotlib.use('Agg')
        from perception.visualization.environment import plot_environment
        figure = plot_environment(snapshot)
        if args.image:
            figure.savefig(args.image,dpi=130)
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == '__main__':
    main()
