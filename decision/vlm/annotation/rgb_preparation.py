"""Prepare perception image associations and register the selected annotation input."""
import os
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from .rules import dumps, require
from .store import Store, database


ROOT = Path(__file__).resolve().parents[3]


def checked_cache(output, identity, assets, expected):
    from perception.detection_tracking.inference.rgb_cache import RGBCache
    cache = RGBCache(output,identity,assets)
    with sqlite3.connect(cache.database.as_uri()+'?mode=ro',uri=True) as db:
        actual = {r[0] for r in db.execute('SELECT frame_id FROM observations')}
    require(expected == actual, 'RGB关联未覆盖工程全部观察，不能登记为已就绪')
    return cache


def prepare_rgb(project, output=None, device='0', *, recompute=False):
    project = Path(project).expanduser().resolve()
    # Producers must be able to resume a project whose selected RGB cache is pending.
    store = Store(project,use_project_defaults=False)
    try:
        identity,assets = store.identity,store.asset_identity
        with database(project/'observations.sqlite',True) as db:
            expected = {r['id'] for r in db.execute('SELECT id FROM frames')}
            previous = db.execute('SELECT value FROM meta WHERE key=?',('default_rgb_associations',)).fetchone()
        registered = json.loads(previous['value']) if previous else None
        output = Path(output).expanduser().resolve() if output else (
            (project/registered['path']).resolve() if registered else project/'rgb_association')
        require(output != project and output not in project.parents, 'RGB输出不能覆盖标注工程')
        require(bool(expected), '观察工程为空')
        recorded=[bool(s.get('rgb_recording')) for s in store.sources()]
    finally:
        store.close()
    manifest = output/'manifest.json'
    if not recompute and manifest.is_file() and json.loads(manifest.read_text()).get('complete') is True:
        # Validate before changing the selected source; a foreign cache must leave it intact.
        cache = checked_cache(output,identity,assets,expected)
        print('登记已有完整RGB关联，不加载YOLO或GPU…',flush=True)
        return select_cache(project,output,cache,len(expected))
    selected = dict(status='pending',path=os.path.relpath(output,project))
    if not registered or registered.get('status') != 'ready' or (project/registered['path']).resolve() == output:
        with database(project/'observations.sqlite') as db:
            db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',('default_rgb_associations',dumps(selected)))
            db.commit()
    command = [sys.executable,'-B',str(ROOT/'perception/scripts/prepare_rgb_association.py'),
               '--project',str(project),'--output',str(output),'--device',str(device)]
    if any(recorded) and not recompute:
        print('导入感知已保存的二维结果，不加载YOLO或GPU…',flush=True)
        from perception.detection_tracking.inference.rgb_recording import import_recordings
        producer=Store(project,use_project_defaults=False)
        try:import_recordings(producer,output)
        finally:producer.close()
    else:
        print('准备YOLO人物框与三维关联；完成后登记为工程默认输入…',flush=True)
        # Share this process group so the unified caller can cancel both preparation and GPU work.
        subprocess.run(command,cwd=ROOT,check=True)
    cache = checked_cache(output,identity,assets,expected)
    return select_cache(project,output,cache,len(expected))


def select_cache(project,output,cache,count):
    selected = dict(status='ready',path=os.path.relpath(output,project),identity=cache.identity)
    with database(project/'observations.sqlite') as db:
        db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',('default_rgb_associations',dumps(selected)))
        db.commit()
    print('决策标注工程已就绪；open／serve／export将自动使用登记的YOLO关联：'+str(project),flush=True)
    return dict(project=str(project),rgb_associations=str(output),identity=cache.identity,observations=count)
