"""Read-only static evidence, cropped in its native frame and bound to a scene."""
import copy
import hashlib
import json

import numpy as np

from perception.detection_tracking.inference.geometry import validate_transform
from .context import crop_map, fingerprint, layer_binding
from .layers import GridSnapshot
from .storage import EnvironmentMap


def map_digest(source):
    # Same asset identity used by the localization authority (not a snapshot ID).
    return hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


class PriorMap:
    def __init__(self, directory, *, expected_map=None):
        package = EnvironmentMap(directory)
        try:
            self.manifest = copy.deepcopy(package.manifest)
            self.snapshot = package.snapshot
            if (self.manifest['kind'] != 'static_environment' or
                    self.manifest['time_scope'] != 'offline_final_prior' or
                    self.snapshot.metadata['frame_id'] != self.manifest['frame_id']):
                raise ValueError('Prior view requires a final static environment package')
            source = self.manifest['source']['map_fingerprint']
            if expected_map is not None and source != expected_map:
                raise ValueError('Static environment belongs to another localization map')
            self.asset = map_digest(source)
            self.observations = [json.loads(row[0]) for row in package.db.execute(
                'SELECT record_json FROM observations ORDER BY stamp_s')]
            for item in self.observations:
                if item['frame_id'] != self.manifest['frame_id']:
                    raise ValueError('Static observation frame disagrees with package')
            self.positions = [validate_transform(item['world_from_lidar'])[:3,3].copy() for item in self.observations]
        finally:
            package.close()

    def attach(self, bundle, settings):
        """Create a new input binding; original scenes and their fingerprints stay intact."""
        from .scene import SceneInputs
        bundle.verify()
        row = copy.deepcopy(bundle.record)
        association = row['prior']
        status = dict(status='unavailable', reason='association_unavailable',
                      time_scope='offline_final_prior')
        grid = None
        measured = association.get('measurement_stamp_s')
        valid = (row['localization']['status'] == 'valid' and association.get('valid') and
            association.get('status') == 'PRIOR_ASSOCIATED' and
            (association.get('local_session_id'), association.get('local_generation'), association.get('local_frame')) ==
            (row['session_id'], row['generation'], row['frame_id']) and
            association.get('stamp_s') == row['as_of'] and measured is not None and
            0 <= row['as_of']-measured <= settings['max_association_age_s'])
        if valid:
            if (association.get('asset_fingerprint') != self.asset or
                    association.get('map_id') != 'roll:'+self.asset or
                    association.get('map_frame') != self.manifest['frame_id']):
                raise ValueError('Prior package asset/frame does not match scene association')
            transform = validate_transform(association['map_from_local'])
            map_pose = transform @ validate_transform(row['localization']['world_from_lidar'])
            radius = float(settings['prior_radius_m'])
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError('Prior radius must be positive and finite')
            # Bound work before copying any raster. Never flatten SE(3) into yaw.
            if (2*radius/self.snapshot.metadata['resolution']+2)**2 > settings['prior_max_cells']:
                raise ValueError('Prior view exceeds configured raster budget')
            bounds = np.r_[map_pose[:2,3]-radius, map_pose[:2,3]+radius]
            meta = self.snapshot.metadata
            low = np.asarray(meta['origin_xy'])
            high = low+np.asarray(meta['shape'])[::-1]*meta['resolution']
            if np.any(bounds[2:] <= low) or np.any(bounds[:2] >= high):
                status['reason'] = 'outside_prior_raster'
            else:
                cropped = crop_map(self.snapshot, bounds, frame=self.manifest['frame_id'],
                                   layers=list(self.snapshot.layers))
                metadata = copy.deepcopy(meta)
                metadata.update(origin_xy=cropped['origin_xy'], shape=list(cropped['layers']['occupancy'].shape),
                    source_snapshot_id=meta['snapshot_id'], view_as_of=row['as_of'],
                    map_from_local=transform.tolist(), map_from_lidar=map_pose.tolist(),
                    association_session=association['association_session'],
                    association_generation=association['association_generation'])
                metadata['snapshot_id'] = 'prior_view:'+fingerprint(dict(source=meta['snapshot_id'],
                    scene=row['input_fingerprint'],bounds=bounds.tolist()))
                for array in cropped['layers'].values():
                    array.flags.writeable = False
                grid = GridSnapshot(metadata, cropped['layers'])
                nearby = []
                for item,position in zip(self.observations,self.positions):
                    distance = float(np.linalg.norm(position[:2]-map_pose[:2,3]))
                    if distance <= radius:
                        nearby.append((distance,item,position))
                refs = [dict(id=item['id'],stamp_s=item['stamp_s'],frame_id=item['frame_id'],
                    position=position.tolist(),distance_m=distance,image_sha256=item['image_sha256'])
                    for distance,item,position in sorted(nearby,key=lambda pair:(pair[0],pair[1]['id']))[:settings['history_limit']]]
                status = dict(status='valid', time_scope='offline_final_prior', source_snapshot_id=meta['snapshot_id'],
                    package_layers_sha256=self.manifest['layers_sha256'], asset_fingerprint=self.asset,
                    frame_id=self.manifest['frame_id'], stamp_s=meta['stamp_s'], as_of=row['as_of'],
                    origin_xy=metadata['origin_xy'], shape=metadata['shape'], resolution=meta['resolution'],
                    map_from_lidar=map_pose.tolist(), map_from_local=transform.tolist(),
                    association_session=association['association_session'],
                    association_generation=association['association_generation'],
                    layer_binding=layer_binding(grid), observations=refs,
                    scope='historical_evidence_not_current_free_space_or_reachability')
        row['prior_geometry'] = status
        row.pop('input_fingerprint')
        row['input_fingerprint'] = fingerprint(row)
        return SceneInputs(row,bundle.geometry,bundle.rgb,grid)


def live_prior_status(bundle, pose, association, now, max_age, max_association_age):
    """A new failure/reassociation immediately withdraws an older displayed prior."""
    from .scene import local_identity
    reason = None
    view = bundle.record.get('prior_geometry',{}) if bundle else {}
    if view.get('status') != 'valid':
        reason = view.get('reason','prior_not_enabled')
    elif pose is None or local_identity(pose) != local_identity(bundle.record) or pose['health'] != 0:
        reason = 'current_localization_unavailable'
    elif not 0 <= now-bundle.record['as_of'] <= max_age:
        reason = 'scene_future_or_stale'
    elif association is None or not association.get('valid'):
        reason = 'current_association_unavailable'
    elif association['stamp_s'] > now or association['stamp_s'] < bundle.record['as_of']:
        reason = 'current_association_time_mismatch'
    elif (association.get('measurement_stamp_s') is None or
          not 0 <= now-association['measurement_stamp_s'] <= max_association_age):
        reason = 'current_association_future_or_stale'
    elif any(association.get(key) != bundle.record['prior'].get(key) for key in (
            'map_id','asset_fingerprint','map_frame','local_frame','local_session_id','local_generation',
            'association_session','association_generation','coverage_fingerprint')):
        reason = 'association_binding_changed'
    return dict(ready=reason is None,reason=reason)
