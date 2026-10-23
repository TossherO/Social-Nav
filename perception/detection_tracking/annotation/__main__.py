"""Detect all bag scans, annotate pedestrian XY centers or export reviews; never starts training."""
import argparse
import json
from pathlib import Path
import sys

from .data import Store, detector_settings, resolve_sequence


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--project', type=Path)
    common.add_argument('--bag', action='append', type=Path, help='新建工程或显式添加来源；可重复')
    common.add_argument('--lidar-topic', default='/rslidar_points')
    common.add_argument('--rgb-topic', default='/camera/color/image_raw', help='空字符串禁用图像')
    common.add_argument('--config', type=Path, help='新工程的检测公共策略覆盖')
    common.add_argument('--checkpoint', type=Path, help='新工程的可信检测权重')
    common.add_argument('--root-group', default='', help='原始采集组；同一原包切片填写相同标识，只影响导出划分')
    common.add_argument('--location-group', default='', help='重复地点组，只影响导出划分，导出前填写')
    common.add_argument('--split', choices=('unassigned', 'train', 'val', 'holdout'), default='unassigned')
    commands = parser.add_subparsers(dest='command', required=True)
    view = commands.add_parser('open', parents=[common], help='按子 bag 浏览缓存／手工标注，不启动检测')
    view.add_argument('--server', help='远程标注HTTP(S)地址；与--project互斥')
    view.add_argument('--project-id', help='服务器登记的工程ID，不是Linux路径')
    view.add_argument('--cache-dir', type=Path, help='客户端缓存目录；默认使用当前用户本地应用数据目录')
    view.add_argument('--read-only', action='store_true', help='只读浏览，不申请写入会话')
    view.add_argument('--image-encoding', choices=('jpeg', 'png'), default='jpeg')
    view.add_argument('--sequence', help='初始打开的子 bag 文件名、完整路径或来源ID')
    view.add_argument('--sample-interval', type=float, help='采样步长秒；省略则恢复该子包上次设置，初始为1秒')
    view.add_argument('--cloud-renderer', choices=('auto', 'software'), default='auto', help='默认 OpenGL；远程桌面不兼容时显式选择 software')
    view.add_argument('--start', type=float, default=0., help='相对各子包首帧点云记录时间的偏移秒')
    view.add_argument('--duration', type=float)
    view.add_argument('--detections', type=Path, help='显式导入带模型来源的既有 observations.jsonl')
    predict = commands.add_parser('predict', parents=[common], help='顺序检测每一扫描，保留原始约10Hz，不启动窗口')
    predict.add_argument('--detector-python', type=Path, required=True, help='既有检测环境解释器')
    predict.add_argument('--gpu', default='0', help='一个可用的物理 GPU 编号')
    export = commands.add_parser('export', help='导出待标／已标帧的核验中心、当前RGB和原始历史点云，不启动训练')
    export.add_argument('--project', required=True, type=Path)
    export.add_argument('--out', type=Path, help='导出目录；默认仓库data/datasets/wheelchair_det_extra/<工程名>，已有目录不覆盖')
    export.add_argument('--complete-only', action='store_true', help='只取已标帧；默认取待标和已标')
    export.add_argument('--history-frames', type=int, default=0, help='每个样本需要的历史点云数，不含当前帧；默认0')
    export.add_argument('--history-stride', type=int, default=1, help='历史采样相隔的原始扫描数；默认1')
    export.add_argument('--pose-results', type=Path, help='已有定位结果根目录，按子包优先localization，其次odometry；也可指定单次运行目录')
    export.add_argument('--max-pose-age', type=float, default=.02, help='只使用过去有效雷达位姿，最大时差秒；默认0.02')
    export.add_argument('--sequence', action='append', help='只导出指定子包；可重复，省略取全部')
    export.add_argument('--start', type=float, default=0., help='相对各子包首帧点云记录时间的偏移秒')
    export.add_argument('--duration', type=float, help='只选择此时长内的标注当前帧；历史可向前越过起点')
    export.add_argument('--allow-missing-images', action='store_true', help='显式保留缺RGB样本并记录image=null；默认缺图报错')
    export.add_argument('--max-image-age', type=float, default=.1, help='只匹配早于当前点云的RGB，最大时差秒；默认0.1')
    export.add_argument('--dry-run', action='store_true', help='只读规划样本和去重后的点云帧数，不读取bag或创建目录')
    service = commands.add_parser('serve', help='启动登记工程的数据服务，不运行检测')
    service.add_argument('--config', required=True, type=Path)
    service.add_argument('--host', help='回环或Tailscale接口IP；默认127.0.0.1')
    service.add_argument('--port', type=int, default=8765)
    package = commands.add_parser('package-client', help='按固定清单生成轻量Windows客户端包')
    package.add_argument('--out', type=Path, required=True)
    drafts = commands.add_parser('export-drafts', help='显式导出本机待同步与冲突存档，不修改服务器')
    drafts.add_argument('--cache-dir', type=Path, required=True)
    drafts.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'open':
        if args.server:
            if not args.project_id or any((args.project, args.bag, args.config, args.checkpoint, args.detections,
                                           args.root_group, args.location_group, args.split != 'unassigned')):
                parser.error('远程打开需要--server和--project-id，不接受本机工程、bag、模型或分组配置')
        elif args.project is None or args.project_id or args.cache_dir or args.read_only:
            parser.error('本机打开需要--project；远程选项须同时指定--server和--project-id')
        import math
        if (args.sample_interval is not None and (not math.isfinite(args.sample_interval) or args.sample_interval <= 0) or
                not math.isfinite(args.start) or args.start < 0 or
                args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0)):
            parser.error('采样间隔／时长须为正有限值，起始偏移须非负')
    if args.command == 'predict':
        if args.project is None:
            parser.error('predict需要--project')
        if not args.gpu.isdigit():
            parser.error('--gpu 只接受一个物理设备编号')
        if not args.detector_python.is_file():
            parser.error('检测解释器不存在')
    if args.command == 'export' and args.out is None:
        args.out = (Path(__file__).resolve().parents[3] / 'data/datasets/wheelchair_det_extra' /
                    args.project.expanduser().resolve().name)
    return args


def main(argv=None):
    args = arguments(argv)
    if args.command == 'serve':
        from .service import serve
        serve(args)
        return 0
    if args.command == 'package-client':
        from .package_client import package_client
        print(package_client(args.out))
        return 0
    if args.command == 'export-drafts':
        from .cache import Cache
        from .data import safe_directory
        if not (args.cache_dir / 'cache.sqlite').is_file():
            raise FileNotFoundError('原客户端缓存不存在；未创建空缓存或导出文件')
        safe_directory(args.out.parent)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(Cache(args.cache_dir).drafts(), stream, ensure_ascii=False, indent=2)
        return 0
    if args.command == 'open' and args.server:
        from .remote import RemoteBackend
        from .app import run
        backend = RemoteBackend(args.server, args.project_id,
                                args.cache_dir, args.read_only, args.image_encoding)
        return run(args, backend=backend)
    if args.command == 'export':
        from .export import export_project, plan_export
        options = dict(complete_only=args.complete_only, history_frames=args.history_frames,
                       history_stride=args.history_stride, sequences=args.sequence,
                       start=args.start, duration=args.duration,
                       pose_results=args.pose_results, max_pose_age_s=args.max_pose_age)
        if args.dry_run:
            result = plan_export(args.project, **options)['counts']
        else:
            result = export_project(args.project, args.out, print, allow_missing_images=args.allow_missing_images,
                                    max_image_age_s=args.max_image_age, **options)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    existing = (args.project / 'annotations.sqlite').is_file()
    for bag in args.bag or []:
        if not bag.is_file():
            raise FileNotFoundError('bag 不存在：' + str(bag))
    if not existing:
        if not args.bag:
            raise ValueError('新建工程需要 --bag；继续工程只需 --project')
        settings, effective = detector_settings(args.config, args.checkpoint)
        store = Store(args.project, create=True, settings=settings, effective=effective)
    else:
        if args.config or args.checkpoint:
            raise ValueError('已有工程保留原配置；切换模型配置请新建工程')
        store = Store(args.project)
    try:
        for bag in args.bag or []:
            store.add_bag(bag, args.lidar_topic, args.rgb_topic,
                          args.root_group, args.location_group, args.split)
        if args.command == 'open' and args.sequence:
            args.sequence = resolve_sequence(store.sources(), args.sequence)
    finally:
        store.close()
    if args.command == 'predict':
        from .offline import predict_project
        print(json.dumps(predict_project(args.project, args.detector_python, args.gpu), ensure_ascii=False, indent=2))
        return 0
    from .app import run
    return run(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        sys.exit(1)
