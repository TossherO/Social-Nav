"""Interactively bind survey starts and build one chronological static map."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from perception.mapping_localization.coverage import full_map_coverage
from perception.mapping_localization.results import file_fingerprint, map_fingerprint
from .scans import sealed_scan_summary

ROOT = Path(__file__).resolve().parents[2]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', required=True, help='Final ROLL localization map')
    parser.add_argument('--evidence-run', action='append', required=True, help='Recorded mapping run; repeat for each bag')
    parser.add_argument('--out', required=True, help='New run directory under outputs/ or /tmp/')
    parser.add_argument('--gpu', help='CUDA device selection for reference localization')
    parser.add_argument('--resolution', type=float, default=.1, help='Static map resolution in metres')
    parser.add_argument('--auto-initial-poses', action='store_true',
                        help='Reuse source-bound historical initial guesses only when map ancestry is verified')
    return parser.parse_args(argv)


def _ancestor_map(target, expected_fingerprint):
    """Verify the update lineage and unchanged parent keyframe coordinate poses."""
    import hashlib
    import numpy as np
    import yaml
    current=Path(target).resolve(); seen=set()
    while current not in seen:
        seen.add(current)
        if map_fingerprint(current) == expected_fingerprint:
            return current
        manifest=yaml.safe_load((current/'manifest.yaml').read_text())
        source_file=current/'update_source.json'
        if manifest.get('source') != 'local_bag_update_candidate' or not source_file.is_file():
            break
        payload=source_file.read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest.get('update_source_sha256'):
            raise ValueError('Updated map ancestry metadata changed')
        source=json.loads(payload);binding=source['old_map']
        parent=(ROOT/binding['path'] if binding['base']=='project_root' else Path(binding['path'])).resolve()
        if map_fingerprint(parent) != source['old_binding']['map_fingerprint']:
            raise ValueError('An ancestor map no longer matches the recorded update source')
        parent_manifest=yaml.safe_load((parent/'manifest.yaml').read_text())
        if parent_manifest.get('frame') != manifest.get('frame'):
            raise ValueError('Updated map ancestry changes coordinate frame')
        before=np.loadtxt(parent/'keyframes/poses.txt',ndmin=2)
        after=np.loadtxt(current/'keyframes/poses.txt',ndmin=2)
        if after.shape[0]<before.shape[0] or after.shape[1]!=before.shape[1] or not np.array_equal(after[:len(before)],before):
            raise ValueError('Updated map ancestry does not preserve parent keyframe poses')
        current=parent
    raise ValueError('Historical initial pose map is not a verified ancestor of the target map')


def automatic_guess(item, target):
    """Return a historical initial guess, never a registered pose or accepted result."""
    from perception.mapping_localization.initial_pose import load_seed
    from .storage import asset_reference
    path=item['path']; run=json.loads((path/'run.json').read_text())
    seed=path.parent/'initial_pose.json'
    if seed.is_file():
        record=json.loads(seed.read_text())
        ancestor=_ancestor_map(target,record['map']['fingerprint'])
        values=load_seed(seed,ancestor,item['bag'],item['offset'])
        return values,dict(method='historical_manual_seed',seed=asset_reference(seed),
                           seed_fingerprint=file_fingerprint(seed),source_map=asset_reference(ancestor),
                           registration_verified=False)
    source_map=path/'map'
    ancestor=_ancestor_map(target,map_fingerprint(source_map))
    values=run['algorithm_parameters']['roll'].get('initialGuess')
    if (not isinstance(values,list) or len(values)!=6 or
            any(type(v) not in (int,float) or not math.isfinite(v) for v in values)):
        raise ValueError('Source mapping run has no valid historical initial guess')
    return values,dict(method='historical_mapping_initial_guess',source_run=asset_reference(path),
                       source_map=asset_reference(ancestor),registration_verified=False)


def prepare(args):
    if not math.isfinite(args.resolution) or args.resolution <= 0:
        raise ValueError('Resolution must be finite and positive')
    target = Path(args.map).expanduser().resolve()
    if not (target/'COMPLETE').is_file():
        raise ValueError('Select a complete ROLL map')
    full_map_coverage(target)
    if len(args.evidence_run) < 2:
        raise ValueError('Provide at least two --evidence-run arguments')
    runs=[]
    for value in args.evidence_run:
        path=Path(value).expanduser().resolve()
        run=json.loads((path/'run.json').read_text())
        if not run.get('success') or run.get('source') != 'bag' or run.get('mode') != 'mapping':
            raise ValueError('Each evidence run must be a successful bag mapping run: '+str(path))
        bag=Path(run['bag']).resolve()
        if file_fingerprint(bag) != run['playback']['bag_fingerprint']:
            raise ValueError('Bag no longer matches evidence run: '+str(path))
        scans=path/'environment/scans.sqlite'
        sealed_scan_summary(scans)
        if file_fingerprint(scans) != run.get('environment_scans_fingerprint'):
            raise ValueError('Sealed scans no longer match evidence run: '+str(path))
        with (path/'localization.jsonl').open() as stream:
            rows=[json.loads(line)['stamp_s'] for line in stream if line.strip()]
        if not rows or any(b <= a for a,b in zip(rows,rows[1:])):
            raise ValueError('Evidence localization times must be strictly increasing: '+str(path))
        runs.append(dict(path=path,bag=bag,start=float(rows[0]),end=float(rows[-1]),offset=float(run.get('start',0.) or 0.),
                         fingerprint=run['playback']['bag_fingerprint']))
    runs.sort(key=lambda row:row['start'])
    if len({json.dumps(row['fingerprint'],sort_keys=True) for row in runs}) != len(runs):
        raise ValueError('Each evidence run must use a distinct bag')
    if any(a['end'] >= b['start'] for a,b in zip(runs,runs[1:])):
        raise ValueError('Survey timestamps overlap; chronological priority cannot be determined')
    requested=Path(args.out).expanduser().absolute()
    if os.path.lexists(requested):
        raise FileExistsError('Refusing existing output: '+str(requested))
    out=requested.resolve()
    if not any(parent in out.parents for parent in (ROOT/'outputs',Path('/tmp').resolve())):
        raise ValueError('Output must be beneath project outputs/ or /tmp/')
    if any(out in item['path'].parents or item['path'] in out.parents or out in item['bag'].parents for item in runs):
        raise ValueError('Output must not overlap an evidence run or input bag')
    if out in target.parents or target in out.parents:
        raise ValueError('Output must not overlap target map')
    return target,runs,out


def run(args):
    target,runs,out=prepare(args)
    out.mkdir(parents=True,exist_ok=False)
    gpu=['--gpu',args.gpu] if args.gpu is not None else []
    print('最终定位地图：'+str(target)+'\n输出目录：'+str(out),flush=True)
    try:
        for i,item in enumerate(runs,1):
            if args.auto_initial_poses:
                values,provenance=automatic_guess(item,target)
                item['pose']=values
                (out/f'initial_guess_{i:02d}.json').write_text(json.dumps(dict(pose_rpy_xyz=values,**provenance),ensure_ascii=False,indent=2)+'\n')
                print(f'\n[{i}/{len(runs)}] 已取得历史初值（尚未验证配准）：{item["bag"]} {values}',flush=True)
            else:
                seed=out/f'initial_pose_{i:02d}.json'
                print(f'\n[{i}/{len(runs)}] 请选择此 bag 的首帧雷达位姿：{item["bag"]}',flush=True)
                subprocess.run(['bash',str(ROOT/'perception/scripts/pick_start.sh'),'--bag',str(item['bag']),
                    '--localization-map',str(target),'--start',str(item['offset']),'--out',str(seed)],cwd=ROOT,check=True)
                if not seed.is_file() or seed.is_symlink():
                    raise RuntimeError('未保存初始位姿，停止处理')
                item['seed']=seed
        command=[sys.executable,'-B','-m','perception.environment','export','--localization-map',str(target)]
        for i,item in enumerate(runs,1):
            reference=out/f'reference_{i:02d}'
            print(f'\n[{i}/{len(runs)}] 重新定位，沿用该采集的传感器与时间参数：{item["path"]}',flush=True)
            pose_args=(['--pose',*map(str,item['pose'])] if 'pose' in item else ['--pose-file',str(item['seed'])])
            subprocess.run(['bash',str(ROOT/'scripts/run.sh'),'--source-run',str(item['path']),
                '--bag',str(item['bag']),'--mode','localization','--map',str(target),
                '--start',str(item['offset']),*pose_args,'--output',str(reference),*gpu],cwd=ROOT,check=True)
            command += ['--evidence-run',str(item['path']),'--reference-run',str(reference)]
        command += ['--resolution',str(args.resolution),'--out',str(out/'static_map'),'--export-pgm']
        print('\n按采集时间融合扫描并导出静态地图。',flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
    except (subprocess.CalledProcessError,OSError,RuntimeError) as error:
        raise RuntimeError(f'处理未完成，已保留中间结果：{out}；请查看上方错误。') from error
    print(f'\n静态地图候选：{out}/static_map\n请检查定位和变化结果，地图不会自动启用。',flush=True)
    return out/'static_map'


def main(argv=None):
    try:
        run(arguments(argv))
    except (ValueError,OSError,RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == '__main__':
    main()
