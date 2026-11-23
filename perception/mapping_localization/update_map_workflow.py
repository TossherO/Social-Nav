"""Select a bag's starting pose and produce reviewable offline map candidates."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess

import yaml

from .coverage import PriorCoverage, asset_digest, full_map_coverage

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT/'navigation/commissioning/interactive.yaml'


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', required=True, help='Read-only local survey bag')
    parser.add_argument('--map', help='Complete old ROLL map; defaults to the interactive configuration')
    parser.add_argument('--coverage', help='Coverage bound to the old map; otherwise use configured or automatic full-map coverage')
    static = parser.add_mutually_exclusive_group()
    static.add_argument('--static-map', help='Old static package bound to the ROLL map; defaults to the interactive configuration when --map is omitted')
    static.add_argument('--3d-only', dest='only_3d', action='store_true', help='Produce only the ROLL candidate')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG), help='Interactive configuration used for map, static map and coverage defaults')
    parser.add_argument('--out', help='New run directory under outputs/ or /tmp/; default is timestamped')
    parser.add_argument('--gpu', help='CUDA_VISIBLE_DEVICES selection for mapping and localization')
    parser.add_argument('--max-voxels', type=int, help='CPU voxel limit for the 3D update; defaults to the static settings')
    args = parser.parse_args(argv)
    if args.max_voxels is not None:
        from .update_map import _validate_max_voxels
        try:
            _validate_max_voxels(args.max_voxels)
        except ValueError as error:
            parser.error(str(error))
    return args


def _configured_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


def prepare(args):
    bag = Path(args.bag).expanduser().resolve()
    if not bag.is_file() or not bag.stat().st_size:
        raise ValueError('Input bag must be an existing nonempty file')
    config = {}
    if args.map is None:
        config = yaml.safe_load(Path(args.config).expanduser().read_text()) or {}
    map_value = args.map if args.map is not None else config.get('map')
    coverage_value = args.coverage if args.coverage is not None else config.get('coverage')
    static_value = args.static_map if args.static_map is not None else config.get('static_map')
    if not map_value:
        raise ValueError('An old map is required; supply --map or configure one')
    old_map = _configured_path(map_value)
    coverage = _configured_path(coverage_value) if coverage_value else None
    old_static = _configured_path(static_value) if static_value and not args.only_3d else None
    if not (old_map/'COMPLETE').is_file() or not (old_map/'GlobalMap.pcd').is_file():
        raise ValueError('Old map must be a complete ROLL map with GlobalMap.pcd')
    if coverage:
        PriorCoverage.load(coverage, asset_fingerprint=asset_digest(old_map))
    else:
        full_map_coverage(old_map)
    default = ROOT/'outputs/perception'/('local_update_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    requested = Path(args.out).expanduser().absolute() if args.out else default
    if os.path.lexists(requested):
        raise FileExistsError('Refusing an existing output: '+str(requested))
    out = requested.resolve()
    if not any(parent in out.parents for parent in (ROOT/'outputs', Path('/tmp').resolve())):
        raise ValueError('Output must be a new directory beneath project outputs/ or /tmp/')
    if out in bag.parents or out in old_map.parents or old_map in out.parents:
        raise ValueError('Output must not contain the input bag or overlap the old map')
    if old_static is not None:
        from perception.environment.incremental import validate_static_parent
        if out == old_static or out in old_static.parents or old_static in out.parents:
            raise ValueError('Output must not overlap the old static map')
        validate_static_parent(old_static, old_map)
    return bag, old_map, coverage, old_static, out


def run(args):
    bag, old_map, coverage, old_static, out = prepare(args)
    out.mkdir(parents=True, exist_ok=False)
    seed, local, reference, candidate = (out/'initial_pose.json', out/'local_mapping',
                                         out/'local_reference', out/'candidate')
    print(f'输入 bag：{bag}\n旧地图：{old_map}\n关联区域：{coverage or "自动全图范围"}\n输出目录：{out}', flush=True)
    if old_static is not None:
        print(f'旧静态地图：{old_static}', flush=True)
    gpu = ['--gpu', args.gpu] if args.gpu is not None else []
    scripts = ROOT/'perception/scripts'
    stage = '人工选择 bag 首帧雷达位姿'
    try:
        print('\n[1/4] '+stage, flush=True)
        subprocess.run(['bash', str(scripts/'pick_start.sh'), '--bag', str(bag),
                        '--localization-map', str(old_map), '--out', str(seed)], check=True, cwd=ROOT)
        if not seed.is_file() or seed.is_symlink():
            raise RuntimeError('未保存起点，已取消后续处理。')
        stage = '局部建图'
        print('\n[2/4] '+stage, flush=True)
        # Local updates are registered to an existing map; independently
        # levelling the short local run would impose an unrelated floor frame.
        subprocess.run(['bash', str(scripts/'build_map.sh'), '--pipeline', 'raw', '--bag', str(bag), '--out', str(local),
                        *gpu], check=True, cwd=ROOT)
        stage = '在旧图中定位'
        print('\n[3/4] '+stage, flush=True)
        subprocess.run(['bash', str(ROOT/'scripts/run.sh'), '--bag', str(bag), '--mode', 'localization',
                        '--source-run', str(local),
                        '--map', str(old_map), '--pose-file', str(seed), *(['--coverage', str(coverage)] if coverage else []),
                        '--output', str(reference), *gpu], check=True, cwd=ROOT)
        stage = '生成候选地图'
        print('\n[4/4] '+stage, flush=True)
        subprocess.run([os.environ.get('SOCIAL_NAV_PYTHON', os.sys.executable), '-B', '-m',
                        'perception.mapping_localization.update_map', '--old-map', str(old_map),
                        '--mapping-run', str(local), '--localization-run', str(reference),
                        *(['--static-map', str(old_static)] if old_static is not None else []),
                        *(['--max-voxels', str(args.max_voxels)] if args.max_voxels is not None else []),
                        '--out', str(candidate)], check=True, cwd=ROOT)
    except (subprocess.CalledProcessError, OSError, RuntimeError) as error:
        raise RuntimeError(f'{stage}未完成；已保留中间结果：{out}。请查看上方错误及相应运行目录。') from error
    print(f'\n候选已生成：{candidate}/map\n查看变化：{candidate}/comparison.png\n'
          f'详细报告：{candidate}/report.json', flush=True)
    if old_static is not None:
        print(f'静态地图候选：{candidate}/static_map\n静态更新报告：{candidate}/static_map/update_report.json', flush=True)
    print('候选不会自动启用，请先核查配准与变化。', flush=True)
    return candidate


def main(argv=None):
    try:
        run(arguments(argv))
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == '__main__':
    main()
