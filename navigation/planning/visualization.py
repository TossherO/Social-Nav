"""Read-only path review; original scans are display evidence, never free rays."""
import base64
import html
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from .backend import ROOT
from .footprint import VehicleGeometry
from .inputs import load_config, load_input
from .planner import se2, transform_poses
from .replay import resolve, write_json


def draw_grid(ax, grid):
    from matplotlib.colors import ListedColormap
    m = grid.metadata; r = m['resolution']; h, w = m['shape']; x, y = m['origin_xy']
    image = np.where(grid.layers['occupancy'] < 0, 0, np.where(grid.layers['occupancy'] == 0, 1, 2))
    if 'conditional_obstacle' in grid.layers:image[grid.layers['conditional_obstacle']]=3
    ax.imshow(image, origin='lower', extent=[x, x+w*r, y, y+h*r], vmin=0, vmax=3,
              cmap=ListedColormap(['#b8c4cf', '#ffffff', '#243e51','#e7a343']), interpolation='nearest')
    ax.set_aspect('equal'); ax.set_xlabel(m['frame_id']+' x (m)'); ax.set_ylabel('y (m)')


def extract_scan(scene_path, run, target):
    from perception.visualization.viewer import BagScans
    from perception.mapping_localization.results import load_result
    from perception.visualization.scans import rotation_matrix
    scene, _ = load_input(scene=scene_path)
    record = scene.record; result = load_result(run)
    matching = [r for r in result.rows if abs(r['stamp_s']-record['as_of']) < 1e-6]
    if len(matching) != 1: raise ValueError('No exact recorded localization row for the scene')
    row = matching[0]
    if (row['session_id'], row['generation'], row['frame_id']) != (record['session_id'], record['generation'], record['frame_id']):
        raise ValueError('Scan source localization lifecycle mismatch')
    position, quaternion = result.pose(row, 'odom')
    transform = np.asarray(record['localization']['world_from_lidar'])
    if not np.allclose(position, transform[:3, 3], atol=1e-5) or not np.allclose(rotation_matrix(quaternion), transform[:3, :3], atol=1e-5):
        raise ValueError('Recorded scan and scene poses disagree')
    scans = BagScans(result.metadata['bag'], result.metadata['playback']['bag_fingerprint'], '/rslidar_points')
    try: points = scans.read(row)
    finally: scans.close()
    if points is None: raise ValueError('Exact raw scan missing')
    points = points @ transform[:3, :3].T+transform[:3, 3]
    trace = [r['position'] for r in result.rows if r['stamp_s'] <= record['as_of'] and r.get('health') == 0
             and (r.get('session_id'), r.get('generation')) == (record['session_id'], record['generation'])]
    np.savez(target, points=points[::max(1, (len(points)+5999)//6000)], trajectory=np.asarray(trace).reshape(-1, 3))


def display_scan(source):
    if not source.get('run') or not source.get('scene'): return None
    scene, _ = load_input(scene=resolve(source['scene']))
    if scene.record['localization']['status'] != 'valid': return None
    environment = os.environ.copy()
    environment['PYTHONNOUSERSITE'] = '1'
    environment['PYTHONPATH'] = str(ROOT)+':/opt/ros/noetic/lib/python3/dist-packages'
    environment['LD_LIBRARY_PATH'] = '/opt/ros/noetic/lib:'+environment.get('LD_LIBRARY_PATH', '')
    with tempfile.TemporaryDirectory(prefix='navigation-scan-') as directory:
        target = Path(directory)/'scan.npz'
        subprocess.run(['/usr/bin/python3', '-B', '-m', __name__, str(resolve(source['scene'])),
                        str(resolve(source['run'])), str(target)], check=True, env=environment, cwd=ROOT,
                       stdout=subprocess.DEVNULL, timeout=45)
        with np.load(target, allow_pickle=False) as data: return {k:data[k] for k in data.files}


def figure_for(row, geometry, *, scan=None):
    import matplotlib.pyplot as plt
    source = row['source']; result = row['result']
    scene, asset = load_input(scene=resolve(source['scene']) if source.get('scene') else None,
                             static_map=resolve(source['static_map']) if source.get('static_map') else None)
    count = (3 if scene.rgb is not None else 2) if scene is not None else 1
    figure, axes = plt.subplots(1, count, figsize=(7*count, 7), squeeze=False, constrained_layout=True)
    axes = axes[0]; global_ax = axes[0]
    grid = asset.snapshot if asset is not None else scene.geometry
    draw_grid(global_ax, grid)
    path = np.asarray(result.get('path', []))
    if len(path):
        color = '#126bd1' if result['current_path_valid'] else '#ad7812'
        global_ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2, label='reference path')
    if 'start' in result:
        p = geometry.polygon(result['start']); global_ax.fill(p[:, 0], p[:, 1], color='#ed621d', alpha=.7)
    request = result['request']; goal = request['goal']
    if request['frame_id'] == grid.metadata['frame_id']: global_ax.plot(*goal[:2], '*', color='#b51f70', markersize=12)
    # Crop the large unknown border for inspection, retaining all observed cells.
    yy, xx = np.where(grid.layers['occupancy'] != -1)
    if len(xx):
        lo = np.array([xx.min(), yy.min()])*grid.metadata['resolution']+grid.metadata['origin_xy']-1
        hi = (np.array([xx.max(), yy.max()])+1)*grid.metadata['resolution']+grid.metadata['origin_xy']+1
        global_ax.set_xlim(lo[0], hi[0]); global_ax.set_ylim(lo[1], hi[1])
    global_ax.set_title('Historical prior | orange: conditional obstacle' if asset is not None else 'Local geometry')
    if scene is not None:
        local_ax = axes[1]; draw_grid(local_ax, scene.geometry)
        pose = (geometry.pose(scene.record['localization']['world_from_lidar'])
                if scene.record['localization']['status'] == 'valid' else None)
        if pose is not None:
            polygon = geometry.polygon(pose); local_ax.fill(polygon[:, 0], polygon[:, 1], color='#ed621d', alpha=.8)
            margin=result.get('safety_margin_m',0.)
            if margin>0:
                from scipy.spatial import ConvexHull
                expanded=(polygon[:,None,:]+margin*np.array([[-1,-1],[-1,1],[1,-1],[1,1]])).reshape(-1,2)
                expanded=expanded[ConvexHull(expanded).vertices]
                expanded=np.vstack([expanded,expanded[0]])
                local_ax.plot(expanded[:,0],expanded[:,1],color='#b55218',linestyle='--',linewidth=1.4)
        if len(path) and pose is not None:
            local = path if request['mode'] == 'local' else transform_poses(path, se2(pose) @ np.linalg.inv(se2(result['start'])))
            local_ax.plot(local[:, 0], local[:, 1], color='#126bd1' if result['current_path_valid'] else '#ad7812', linewidth=1.8)
            local_ax.plot(*local[-1, :2], '*', color='#b51f70', markersize=10)
            check = result.get('local_check', {})
            if check.get('last_valid_pose'):
                last = np.asarray(check['last_valid_pose'])
                color='#ad7812' if result.get('current_local_path',{}).get('source')=='local_replan' else '#24a246'
                local_ax.plot(*last[:2], 'o', color=color, markersize=7)
            prefix=np.asarray(result.get('current_local_path',{}).get('poses',[]))
            if len(prefix):
                local_ax.plot(prefix[:,0],prefix[:,1],color='#24a246',linewidth=3)
                if result.get('current_local_path',{}).get('source')=='local_replan':
                    local_ax.plot(*prefix[-1,:2],'*',color='#24a246',markersize=10)
        if scan is not None:
            points = scan['points']; local_ax.scatter(points[:, 0], points[:, 1], s=.8, c='#07a6c6', alpha=.5)
            trace = scan['trajectory']; local_ax.plot(trace[:, 0], trace[:, 1], color='#6f59b6', linewidth=1)
        if pose is not None:
            local_ax.arrow(*pose[:2], .65*np.cos(pose[2]), .65*np.sin(pose[2]), head_width=.15, color='#92380d')
        local_ax.set_xlim(scene.geometry.metadata['origin_xy'][0], scene.geometry.metadata['origin_xy'][0]+scene.geometry.metadata['shape'][1]*scene.geometry.metadata['resolution'])
        local_ax.set_ylim(scene.geometry.metadata['origin_xy'][1], scene.geometry.metadata['origin_xy'][1]+scene.geometry.metadata['shape'][0]*scene.geometry.metadata['resolution'])
        radius=source.get('local_view_radius_m')
        if radius is not None and pose is not None:
            if not np.isfinite(radius) or not 0<radius<=20:raise ValueError('Invalid review zoom radius')
            local_ax.set_xlim(pose[0]-radius,pose[0]+radius);local_ax.set_ylim(pose[1]-radius,pose[1]+radius)
        local_ax.set_title('Current local grid | cyan raw scan | purple causal trace')
        if scene.rgb is not None:
            axes[2].imshow(scene.rgb)
            axes[2].axis('off'); axes[2].set_title('Bound RGB; no new inference')
    figure.suptitle('%s | %s | check=%s | local=%s | executable=false' % (
        row['label'],result['status'],result.get('reason') or result.get('local_check',{}).get('status',''),
        result.get('current_local_path',{}).get('source','none')))
    return figure


def review(results_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = Path(results_path); raw=path.read_bytes();data=json.loads(raw);frames=[]
    source_fingerprint=hashlib.sha256(raw).hexdigest()
    cfg = load_config(); geometry = VehicleGeometry.historical(cfg['geometry_reference'])
    scan_cache = {}
    for row in data['queries']:
        if 'result' not in row: continue
        key = (row['source'].get('scene'), row['source'].get('run'))
        if key not in scan_cache: scan_cache[key] = display_scan(row['source'])
        fig = figure_for(row, geometry, scan=scan_cache[key]); stream = io.BytesIO()
        fig.savefig(stream, format='png', dpi=105); plt.close(fig)
        frames.append(dict(label=row['label'], result=row['result'],
                           png='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()))
    destination = path.with_name('review.html')
    if destination.exists(): raise FileExistsError(destination)
    payload = json.dumps(frames, ensure_ascii=False).replace('<', '\\u003c')
    page = '''<!doctype html><meta charset="utf-8"><title>导航规划集中审核</title>
<style>body{font:16px sans-serif;margin:20px;background:#eff3f7}button{padding:8px}img{width:100%;background:white}pre{white-space:pre-wrap}input{width:65%}</style>
<h2>导航规划集中审核</h2><p>灰：未知；白：空闲；深色：障碍；橙：车体假设，虚线为安全余量；蓝：全部适用几何检查通过的路径；土黄：局部未完全验证的参考路线；绿：已验证局部路线（参考前缀或局部重规划）；青：原始扫描；紫：截止当前的定位轨迹。所有结果不可执行。</p>
<button id="prev">上一项</button> <button id="play">播放／暂停</button> <button id="next">下一项</button>
<input id="seek" type="range" min="0" step="1"><span id="label"></span>
<img id="picture"><pre id="info"></pre><h3>审核意见</h3>
<p>REVIEW_PROMPT</p>
<textarea id="answers" rows="8" style="width:98%" placeholder="逐项意见"></textarea><button id="save">下载审核意见</button>
<script>const frames=PAYLOAD;let i=0,timer=null;
const seek=document.getElementById('seek'),pic=document.getElementById('picture');seek.max=frames.length-1;
function show(n){pic.removeAttribute('src');i=Math.max(0,Math.min(frames.length-1,n));seek.value=i;
pic.src=frames[i].png;document.getElementById('label').textContent=(i+1)+'/'+frames.length+' '+frames[i].label;
const r=frames[i].result;document.getElementById('info').textContent=JSON.stringify({status:r.status,reason:r.reason,search_backend:r.search_backend,current_path_valid:r.current_path_valid,current_local_path:r.current_local_path?{valid:r.current_local_path.valid,complete:r.current_local_path.complete,source:r.current_local_path.source}:null,local_replan:r.local_replan?{status:r.local_replan.status,reason:r.local_replan.reason,path_length_m:r.local_replan.path_length_m,goal:r.local_replan.request.goal}:null,needs_observation:r.needs_observation,local_start_evidence:r.local_start_evidence,local_check:r.local_check,timing_ms:r.timing_ms,geometry_assumed:r.geometry_assumed},null,2)}
document.getElementById('prev').onclick=()=>show(i-1);document.getElementById('next').onclick=()=>show(i+1);seek.oninput=()=>show(+seek.value);
document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null}else timer=setInterval(()=>{if(i==frames.length-1){clearInterval(timer);timer=null}else show(i+1)},1800)};
document.getElementById('save').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({source:'RESULTS',source_fingerprint:'SOURCE_FINGERPRINT',case_labels:frames.map(f=>f.label),answers:document.getElementById('answers').value},null,2)],{type:'application/json'}));a.download='navigation_review_answers.json';a.click()};show(0);</script>'''
    prompt=data.get('review_prompt','请集中填写车体假设是否合理、路径是否穿墙／贴障、局部未知拒绝是否符合图示，并记录样本名称。')
    destination.write_text(page.replace('REVIEW_PROMPT',html.escape(prompt)).replace('PAYLOAD',payload)
                           .replace('RESULTS',html.escape(path.name)).replace('SOURCE_FINGERPRINT',source_fingerprint))
    return destination


def pick_endpoints(static_map, output):
    import matplotlib.pyplot as plt
    _, asset = load_input(static_map=static_map)
    fig, ax = plt.subplots(figsize=(10, 9)); draw_grid(ax, asset.snapshot)
    ax.set_title('Pan/zoom first; close toolbar modes; click START then GOAL (yaw=0)')
    points = plt.ginput(2, timeout=0)
    if len(points) != 2: raise ValueError('Two explicit points are required')
    write_json(output, dict(start=[*points[0], 0.], goal=list(points[1]), target_reference='base_link',
                            frame_id=asset.manifest['frame_id'], map_snapshot_id=asset.manifest['snapshot_id']))
    plt.close(fig)


def rviz_view(results_path):
    from .ros_display import run
    return run(results_path)


if __name__ == '__main__': extract_scan(*sys.argv[1:])
