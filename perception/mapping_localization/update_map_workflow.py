"""Select a bag's starting pose and produce a reviewable 3D map update."""
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
    parser.add_argument('--config', default=str(DEFAULT_CONFIG), help='Interactive configuration used for map and coverage defaults')
    parser.add_argument('--out', help='New run directory under outputs/ or /tmp/; default is timestamped')
    parser.add_argument('--gpu', help='CUDA_VISIBLE_DEVICES selection for mapping and localization')
    return parser.parse_args(argv)


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
    if not map_value:
        raise ValueError('An old map is required; supply --map or configure one')
    old_map = _configured_path(map_value)
    coverage = _configured_path(coverage_value) if coverage_value else None
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
    return bag, old_map, coverage, out


def run(args):
    bag, old_map, coverage, out = prepare(args)
    out.mkdir(parents=True, exist_ok=False)
    seed, local, reference, candidate = (out/'initial_pose.json', out/'local_mapping',
                                         out/'local_reference', out/'candidate')
    print(f'输入 bag：{bag}\n旧地图：{old_map}\n关联区域：{coverage or "自动全图范围"}\n输出目录：{out}', flush=True)
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
        subprocess.run(['bash', str(scripts/'build_map.sh'), '--bag', str(bag), '--out', str(local),
                        *gpu], check=True, cwd=ROOT)
        stage = '在旧图中定位'
        print('\n[3/4] '+stage, flush=True)
        subprocess.run(['bash', str(ROOT/'scripts/run.sh'), '--bag', str(bag), '--mode', 'localization',
                        '--map', str(old_map), '--pose-file', str(seed), *(['--coverage', str(coverage)] if coverage else []),
                        '--output', str(reference), *gpu], check=True, cwd=ROOT)
        stage = '生成候选地图'
        print('\n[4/4] '+stage, flush=True)
        subprocess.run([os.environ.get('SOCIAL_NAV_PYTHON', os.sys.executable), '-B', '-m',
                        'perception.mapping_localization.update_map', '--old-map', str(old_map),
                        '--mapping-run', str(local), '--localization-run', str(reference),
                        '--out', str(candidate)], check=True, cwd=ROOT)
    except (subprocess.CalledProcessError, OSError, RuntimeError) as error:
        raise RuntimeError(f'{stage}未完成；已保留中间结果：{out}。请查看上方错误及相应运行目录。') from error
    print(f'\n候选已生成：{candidate}/map\n查看变化：{candidate}/comparison_detail.png\n'
          f'详细报告：{candidate}/report.json\n候选不会自动启用，请先核查配准与变化。', flush=True)
    return candidate


def main(argv=None):
    try:
        run(arguments(argv))
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == '__main__':
    main()
