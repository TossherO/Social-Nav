"""Offline scan-ground evidence must not turn conflicting measurements into support."""
import unittest
from unittest.mock import patch

import numpy as np

from perception.environment.layers import GridSnapshot, load_config
from perception.environment.offline_ground import scan_ground


class SyntheticTracker:
    def __init__(self, _settings):self.observed_ground=None
    def fit(self, cloud, _sensor, _config):
        self.observed_ground=cloud
        return None,dict(status='tracked')


class OfflineGroundTests(unittest.TestCase):
    def test_conflicting_cells_withdraw_support_and_sparse_cells_remain_marked(self):
        config=load_config();config['resolution']=1.;config['ground_support_radius_m']=1.5
        grid=GridSnapshot(dict(shape=[3,3],origin_xy=[0.,0.],resolution=1.,snapshot_id='old',map_id='map'),
            dict(occupancy=np.full((3,3),-1,np.int8),ground_z=np.full((3,3),np.nan,np.float32),
                 ground_support=np.zeros((3,3),bool)))
        observations=[np.array([[.5,.5,0.],[1.5,.5,0.],[.5,1.5,0.],[1.5,1.5,.5]]),
                      np.array([[.5,.5,0.],[1.5,.5,.5]]),
                      np.array([[.5,.5,0.],[1.5,.5,.5]])]
        records=[(dict(stamp_ns=int((1+i*.25)*1e9)),dict(cloud=value)) for i,value in enumerate(observations)]
        def geometry(record,data,_timeline):
            pose=dict(session_id='s',generation=0,segment=1)
            return record['stamp_ns']/1e9,pose,None,np.eye(4),data['cloud'],None
        with patch('perception.environment.offline_ground.sampled_scan_stamps',return_value=(3,[r[0]['stamp_ns'] for r in records])), \
             patch('perception.environment.offline_ground.read_scans',return_value=iter(records)), \
             patch('perception.environment.offline_ground._scan_geometry',side_effect=geometry), \
             patch('perception.environment.offline_ground.GroundTracker',SyntheticTracker):
            result=scan_ground(grid,'unused',None,config,{'source':'synthetic'})
        self.assertTrue(result.layers['ground_support'][0,0])
        self.assertFalse(result.layers['ground_support'][0,1])
        self.assertTrue(result.layers['ground_support'][1,0])
        self.assertTrue(result.layers['ground_scan_undersampled'][1,0])
        self.assertFalse(result.layers['ground_support'][1,1])
        self.assertTrue(result.layers['ground_scan_unreliable'][1,1])
        self.assertTrue(np.isnan(result.layers['ground_z'][0,1]))
        self.assertTrue(result.layers['ground_scan_unreliable'][0,1])
        self.assertEqual(result.metadata['ground_scan_diagnostics']['conflicting_cells'],1)
        self.assertEqual(result.metadata['ground_scan_diagnostics']['undersampled_cells'],2)
        self.assertEqual(result.metadata['ground_scan_diagnostics']['contradictory_sparse_cells'],1)
        self.assertFalse(grid.layers['ground_support'].any())

    def test_missing_ground_never_reuses_legacy_height(self):
        config=load_config()
        grid=GridSnapshot(dict(shape=[1,1],origin_xy=[0.,0.],resolution=.1,snapshot_id='old',map_id='map'),
            dict(occupancy=np.array([[0]],np.int8),ground_z=np.array([[0.]],np.float32),
                 ground_support=np.array([[True]])))
        with patch('perception.environment.offline_ground.sampled_scan_stamps',return_value=(0,[])), \
             patch('perception.environment.offline_ground.read_scans',return_value=iter([])):
            with self.assertRaisesRegex(ValueError,'No tracked scan ground'):
                scan_ground(grid,'unused',None,config,{'source':'synthetic'})

    def test_neighboring_reliable_but_incompatible_surfaces_are_unknown_at_boundary(self):
        config=load_config();config['resolution']=1.;config['ground_support_radius_m']=.1
        grid=GridSnapshot(dict(shape=[1,4],origin_xy=[0.,0.],resolution=1.,snapshot_id='old',map_id='map'),
            dict(occupancy=np.full((1,4),-1,np.int8),ground_z=np.full((1,4),np.nan,np.float32),
                 ground_support=np.zeros((1,4),bool)))
        observations=[np.array([[.5,.5,0.],[1.5,.5,.4],[3.5,.5,0.]]) for _ in range(3)]
        records=[(dict(stamp_ns=int((1+i*.25)*1e9)),dict(cloud=value)) for i,value in enumerate(observations)]
        def geometry(record,data,_timeline):
            return record['stamp_ns']/1e9,dict(session_id='s',generation=0,segment=1),None,np.eye(4),data['cloud'],None
        with patch('perception.environment.offline_ground.sampled_scan_stamps',return_value=(3,[r[0]['stamp_ns'] for r in records])), \
             patch('perception.environment.offline_ground.read_scans',return_value=iter(records)), \
             patch('perception.environment.offline_ground._scan_geometry',side_effect=geometry), \
             patch('perception.environment.offline_ground.GroundTracker',SyntheticTracker):
            result=scan_ground(grid,'unused',None,config,{'source':'synthetic'})
        self.assertFalse(result.layers['ground_support'][0,:2].any())
        self.assertTrue(result.layers['ground_support'][0,3])
        self.assertTrue(result.layers['ground_scan_unreliable'][0,:2].all())
        self.assertEqual(result.metadata['ground_scan_diagnostics']['discontinuous_cells'],2)


if __name__=='__main__':unittest.main()
