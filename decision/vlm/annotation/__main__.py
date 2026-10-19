"""Independent T1/T2 annotation commands; no eager GUI, ROS or model imports."""
import argparse
import json
import math
from pathlib import Path


def arguments(argv=None):
    parser = argparse.ArgumentParser(description='T1/T2决策标注：离线准备、本机／远程界面与标签导出')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='快速导入已有结果与bag头信息，原始帧按需读取')
    prepare.add_argument('--results', nargs='+', required=True, help='感知结果目录，或people.jsonl')
    prepare.add_argument('--bag', nargs='+', help='按results顺序显式定位bag；默认从run.json读取')
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--ros-python', default='/usr/bin/python3')
    prepare.add_argument('--calibration', default=str(Path(__file__).resolve().parents[3] / 'perception/sensors/calibration/front_camera_lidar.yaml'))
    prepare.add_argument('--pointcloud-topic', default='/rslidar_points')
    prepare.add_argument('--rgb-topic', default='/camera/color/image_raw')
    prepare.add_argument('--camera-info-topic', default='/camera/color/camera_info')
    prepare.add_argument('--root-group', help='共同原始采集来源；同一长bag切片须一致')
    prepare.add_argument('--start', type=float, default=0)
    prepare.add_argument('--duration', type=float)
    opened = commands.add_parser('open', help='独立Qt客户端')
    location = opened.add_mutually_exclusive_group(required=True)
    location.add_argument('--project')
    location.add_argument('--server')
    opened.add_argument('--project-id', default='wheelchair-vlm')
    opened.add_argument('--read-only', action='store_true')
    opened.add_argument('--cache-dir')
    served = commands.add_parser('serve', help='Linux HTTP服务；默认只读')
    served.add_argument('--project', required=True)
    served.add_argument('--project-id', default='wheelchair-vlm')
    served.add_argument('--host', default='127.0.0.1')
    served.add_argument('--port', type=int, default=8766)
    served.add_argument('--write', action='store_true')
    exported = commands.add_parser('export', help='仅导出服务器正式已标标签')
    exported.add_argument('--project', required=True)
    exported.add_argument('--output', required=True)
    exported.add_argument('--include-inputs', action='store_true')
    packed = commands.add_parser('package-client', help='按固定清单生成客户端ZIP')
    packed.add_argument('--output', required=True)
    drafts = commands.add_parser('export-drafts', help='离线导出Windows缓存的未同步操作')
    drafts.add_argument('--cache-dir', required=True)
    drafts.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        if not math.isfinite(args.start) or args.start < 0 or (args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0)):
            parser.error('start须非负、duration须为正有限秒数')
    if args.command == 'serve' and not 1 <= args.port <= 65535:
        parser.error('端口错误')
    return args


def main(argv=None):
    args = arguments(argv)
    if args.command == 'prepare':
        from .prepare import prepare
        prepare(args)
    elif args.command == 'serve':
        from .service import serve
        serve(args)
    elif args.command == 'open':
        from .app import run
        run(args)
    elif args.command == 'export':
        from .export import export_project
        print(json.dumps(export_project(args.project, args.output, args.include_inputs), ensure_ascii=False))
    elif args.command == 'package-client':
        from .package_client import package_client
        print('打包文件数：' + str(package_client(args.output)))
    elif args.command == 'export-drafts':
        import sqlite3
        path = Path(args.output)
        if path.exists():
            raise ValueError('导出目标已存在')
        uri = (Path(args.cache_dir) / 'cache.sqlite').resolve().as_uri() + '?mode=ro'
        with sqlite3.connect(uri, uri=True) as db:
            rows = [json.loads(r[0]) for r in db.execute('SELECT request FROM queue ORDER BY seq')]
            archived = [json.loads(r[0]) for r in db.execute('SELECT document FROM archives ORDER BY seq')]
        path.write_text(json.dumps(dict(pending=rows, archives=archived), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print('未同步操作数：' + str(len(rows)))


if __name__ == '__main__':
    main()
