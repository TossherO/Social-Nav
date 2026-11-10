"""Prepare map-bound observation assets in a new outputs directory."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from perception.environment.storage import StaticEnvironmentMap
from perception.mapping_localization.coverage import save_coverage, asset_digest
from perception.mapping_localization.results import map_fingerprint

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True, help='New directory below outputs; never overwrites current commissioning assets')
OUT=Path(parser.parse_args().output).expanduser().resolve()
DATA=ROOT/'outputs/perception/lab_floor2_20261107'
MAP=DATA/'mapping_compensated/map'
STATIC=DATA/'static_scans_10cm_20261110'
SEED=DATA/'navigation_preparation/initial_pose.json'
if not OUT.is_relative_to((ROOT/'outputs').resolve()) or OUT == (ROOT/'outputs').resolve():
    raise ValueError('Output must be a new directory below outputs')
OUT.mkdir(parents=True,exist_ok=False)
asset=StaticEnvironmentMap(STATIC)
fingerprint=map_fingerprint(MAP)
assert (MAP/'COMPLETE').is_file()
assert asset.manifest['source']['map_fingerprint']==fingerprint
seed=json.loads(SEED.read_text());assert seed['map']['fingerprint']==fingerprint
region=[[-12.,-4.],[1.,-4.],[1.,.5],[-12.,.5]]
coverage=OUT/'coverage_observation.yaml'
if coverage.exists():raise FileExistsError(coverage)
save_coverage(coverage,MAP,[region],source=dict(
    kind='developer_selected_candidate_region_for_live_shadow_observation',
    reviewed_source='selected_static_grid_and_offline_first_scan_registration',
    field_verified=False,free_space_assertion=False,execution_accepted=False,
    limitation='Candidate map-association domain only; local obstacle checks remain unchanged. Field review is pending.'))
hint=dict(format='map_height_attitude_hint',source=dict(
    kind='derived_from_offline_registered_navigation_first_scan',path=str(SEED),
    sha256=hashlib.sha256(SEED.read_bytes()).hexdigest()),
    map=seed['map'],pose=seed['pose'],registration_verified=False,
    live_usage='Only roll, pitch and map Z seed; operator must select current LiDAR XY and heading in RViz.')
config=dict(map=str(MAP.relative_to(ROOT)),static_map=str(STATIC.relative_to(ROOT)),
    coverage=str(coverage.relative_to(ROOT)),initial_pose_hint=str((OUT/'height_hint.json').relative_to(ROOT)),
    motion_config='navigation/planning/configs/motion.yaml',motion_profile='trial')
for name,value in [('height_hint.json',hint),('manifest.json',dict(scope='live_shadow_software_preparation',
        actuation_enabled=False,map_field_acceptance=False,coverage_field_acceptance=False,
        map_fingerprint=fingerprint,map_asset_digest=asset_digest(MAP),
        static_snapshot_id=asset.manifest['snapshot_id'],region_xy_m=region,
        limits=['The map retains the previously reported height deformation.',
                'ROI selection is not free-space, reachability, or physical acceptance.',
                'Live XY and yaw are selected anew; no recorded bag pose initializes live input.',
                'Vehicle motion, protocol and firmware protections still require verification.']))]:
    with (OUT/name).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
with (OUT/'interactive.yaml').open('x') as stream:yaml.safe_dump(config,stream,allow_unicode=True,sort_keys=False)
g=asset.snapshot;m=g.metadata;cells=g.layers['occupancy']
extent=[m['origin_xy'][0],m['origin_xy'][0]+cells.shape[1]*m['resolution'],
        m['origin_xy'][1],m['origin_xy'][1]+cells.shape[0]*m['resolution']]
fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
image=np.where(cells==100,0.,np.where(cells==0,1.,.6))
known=g.layers['ground_support']&np.isfinite(g.layers['ground_z'])
for ax in axes:
    ax.imshow(image,origin='lower',extent=extent,cmap='gray',vmin=0,vmax=1)
    polygon=np.array(region+[region[0]])
    ax.plot(polygon[:,0],polygon[:,1],color='tab:orange',lw=2,label='Candidate observation domain')
    ax.scatter(*seed['pose']['position_m'][:2],marker='x',c='red',s=70,label='Offline height hint anchor')
    ax.set_xlim(-13,2);ax.set_ylim(-5,1.5);ax.set_aspect('equal');ax.set_xlabel('map x [m]');ax.set_ylabel('map y [m]')
axes[0].set_title('Static occupancy: white free, gray unknown, black occupied')
im=axes[1].imshow(np.where(known,g.layers['ground_z'],np.nan),origin='lower',extent=extent,cmap='viridis',alpha=.8)
axes[1].set_title('Available static ground heights');fig.colorbar(im,ax=axes[1],label='map ground z [m]')
axes[0].legend(fontsize=8)
fig.suptitle('Observation-only candidate region; field verification pending')
fig.savefig(OUT/'observation_region.png',dpi=140);plt.close(fig);asset.close()
print(json.dumps(config,ensure_ascii=False,indent=2))
