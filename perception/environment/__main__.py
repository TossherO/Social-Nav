"""Inspect geometry, export environment packages and query visual/place memory."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .layers import GridSnapshot
from .context import query_geometry


def bundle_main(argv):
    from .export import export_environment
    from .storage import EnvironmentMap, MemoryReader
    from .layers import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export', help='Build a new offline package from a recorded run')
    export.add_argument('--run', required=True)
    export.add_argument('--out', required=True)
    export.add_argument('--config')
    export.add_argument('--bag', help='Explicit relocation of the same bound source bag')
    export.add_argument('--camera-calibration', help='Explicit same-installation camera/LiDAR YAML')
    export.add_argument('--navigation', action='store_true', help='Also export standard map_server PGM/YAML')
    query = commands.add_parser('query', help='Read one package; never invokes models or robot motion')
    query.add_argument('--map', required=True)
    query.add_argument('--operation', choices=['summary', 'observations', 'places', 'geometry', 'crop'], default='summary')
    query.add_argument('--as-of', type=float, help='Required acquisition-time cutoff for memory queries')
    query.add_argument('--frame')
    query.add_argument('--place')
    query.add_argument('--position', type=float, nargs=2)
    query.add_argument('--radius', type=float)
    query.add_argument('--heading', type=float, help='Radians, optical +Z or explicit LiDAR +X fallback')
    query.add_argument('--limit', type=int, default=20)
    query.add_argument('--bounds', type=float, nargs=4)
    query.add_argument('--layers', nargs='+', default=['occupancy'])
    image = commands.add_parser('image', help='Decode one indexed RGB observation from its original bag')
    image_source = image.add_mutually_exclusive_group(required=True)
    image_source.add_argument('--map')
    image_source.add_argument('--memory', help='Live/replayed run environment directory')
    image.add_argument('--observation', required=True)
    image.add_argument('--bag')
    image.add_argument('--out', required=True)
    memory = commands.add_parser('query-memory', help='Read live/replayed visual memory with explicit local lifecycle')
    memory.add_argument('--memory', required=True)
    memory.add_argument('--as-of', type=float)
    memory.add_argument('--session')
    memory.add_argument('--generation', type=int)
    memory.add_argument('--frame')
    memory.add_argument('--limit', type=int, default=20)
    view = commands.add_parser('view', help='View geometry and up to 256 indexed historical observations')
    view.add_argument('--map', required=True)
    view.add_argument('--as-of', type=float, required=True)
    view.add_argument('--out', help='New overview image')
    view.add_argument('--show', action='store_true')
    place = commands.add_parser('add-place', help='Add one manually named place to an existing package')
    place.add_argument('--map', required=True)
    place.add_argument('--id', required=True)
    place.add_argument('--name', required=True)
    place.add_argument('--observation', required=True)
    place.add_argument('--frame', required=True)
    place.add_argument('--position', type=float, nargs=3)
    place.add_argument('--bounds', type=float, nargs=4)
    place.add_argument('--alias', action='append', default=[])
    args = parser.parse_args(argv)
    if args.command == 'export':
        data = export_environment(args.run, args.out, config=load_config(args.config), bag=args.bag,
                                  calibration=args.camera_calibration, navigation=args.navigation)
        print(json.dumps(dict(out=args.out, kind=data['kind'], images=data['images']), ensure_ascii=False))
        return
    live_memory = args.command == 'query-memory' or (args.command == 'image' and args.memory)
    package = MemoryReader(args.memory) if live_memory else EnvironmentMap(args.map)
    try:
        if args.command == 'query-memory':
            result = package.summary() if args.as_of is None else package.query(dict(
                operation='query_observations',memory_id=package.binding['memory_id'],
                session_id=args.session,generation=args.generation,frame=args.frame,
                as_of=args.as_of,limit=args.limit))
        elif args.command == 'view':
            if not args.out and not args.show:
                parser.error('Choose --out or --show')
            if args.out and Path(args.out).exists():
                raise FileExistsError('Select a new overview image path')
            if not args.show:
                import matplotlib
                matplotlib.use('Agg')
            from perception.visualization.environment import plot_memory
            rows = package.query_observations(as_of=args.as_of, frame=package.manifest['frame_id'], limit=256)
            figure = plot_memory(package.snapshot, rows)
            if args.out:
                figure.savefig(args.out, dpi=130)
            if args.show:
                import matplotlib.pyplot as plt
                plt.show()
            result = dict(out=args.out, displayed_observations=len(rows), as_of=args.as_of)
        elif args.command == 'image':
            target = Path(args.out)
            if target.exists():
                raise FileExistsError('Select a new image path')
            rgb = package.read_image(args.observation, bag_override=args.bag)
            import cv2
            success, data = cv2.imencode(target.suffix, rgb[:, :, ::-1])
            if not success:
                raise ValueError('Unsupported output image format')
            with target.open('xb') as stream:
                stream.write(data.tobytes())
            result = dict(out=str(target), observation_id=args.observation)
        elif args.command == 'add-place':
            result = package.add_place(args.id, args.name, args.observation, frame=args.frame,
                                       position=args.position, bounds=args.bounds, aliases=args.alias)
        elif args.operation == 'summary':
            result = package.summary()
        elif args.operation in ('observations', 'places'):
            if args.as_of is None:
                parser.error('Memory queries require an explicit --as-of acquisition timestamp')
            if args.operation == 'observations':
                result = package.query_observations(as_of=args.as_of, place=args.place, frame=args.frame,
                    position=args.position, radius=args.radius, heading=args.heading, limit=args.limit)
            else:
                result = package.query_places(as_of=args.as_of, name=args.place, bounds=args.bounds, frame=args.frame)
        elif args.operation == 'geometry':
            if args.position is None:
                parser.error('Geometry queries require --position X Y')
            result = package.query_geometry([args.position], frame=args.frame)
        else:
            if args.bounds is None:
                parser.error('Map crops require --bounds xmin ymin xmax ymax')
            result = package.crop_map(args.bounds, frame=args.frame, layers=args.layers)
        from .context import json_safe
        print(json.dumps(json_safe(result), ensure_ascii=False, allow_nan=False))
    finally:
        package.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ('export', 'query', 'image', 'view', 'add-place', 'query-memory'):
        return bundle_main(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.epilog = 'Package commands: export, query, query-memory, image, view, add-place (COMMAND --help).'
    parser.add_argument('--snapshot',required=True)
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
