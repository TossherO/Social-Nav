"""Reference-only evidence projection, with independent polygon overlap checks."""
import unittest
import numpy as np

from perception.environment.layers import GridSnapshot
from navigation.planning.overlay import reference_overlay,overlap_states
from navigation.planning.footprint import VehicleGeometry,polygon_cells
from navigation.planning.inputs import load_config
from navigation.planning.planner import Planner
from tests.navigation_fixtures import asset,bundle,request


class ReferenceOverlayTests(unittest.TestCase):
    def setUp(self):
        self.scene=bundle();self.local=self.scene.geometry
        cells=np.zeros((60,60),np.int8);cells[20,20]=-1;cells[30,30]=100
        self.prior=asset(cells).snapshot

    def apply(self,local=None,transform=None):
        return reference_overlay(self.prior,local or self.local,self.local.layers['occupancy'],
            np.eye(3) if transform is None else transform,self.scene)

    def test_existing_local_free_complements_unknown_without_mutating_evidence(self):
        original=self.prior.layers['occupancy'].copy()
        result,report=self.apply()
        self.assertEqual(result.layers['occupancy'][20,20],0)
        self.assertEqual(result.layers['occupancy'][30,30],100)
        self.assertEqual(report['unknown_to_free'],1)
        self.assertEqual(report['source_as_of'],self.scene.record['as_of'])
        self.assertEqual(report['source_scene'],self.scene.record['input_fingerprint'])
        self.assertFalse(report['persistent']);self.assertNotIn('last_observed',result.layers)
        self.assertFalse(result.layers['occupancy'].flags.writeable)
        np.testing.assert_array_equal(self.prior.layers['occupancy'],original)

    def test_static_derived_free_cannot_recursively_supply_prior_unknowns(self):
        self.local.layers['occupancy'][20,20]=-1
        derived=GridSnapshot(self.local.metadata,dict(occupancy=np.zeros((60,60),np.int8)))
        result,_=self.apply(derived)
        self.assertEqual(result.layers['occupancy'][20,20],-1)

    def test_current_endpoint_overlay_vetoes_free_raw_background(self):
        cells=self.local.layers['occupancy'].copy();cells[20,20]=100
        result,_=self.apply(GridSnapshot(self.local.metadata,dict(occupancy=cells)))
        self.assertEqual(result.layers['occupancy'][20,20],-1)

    def test_free_centre_does_not_hide_unknown_or_occupied_corner(self):
        transform=np.eye(3);transform[0,2]=.03
        for value in (-1,100):
            self.local.layers['occupancy'][20,21]=value
            result,_=self.apply(transform=transform)
            self.assertEqual(result.layers['occupancy'][20,20],-1)

    def test_next_query_withdraws_unavailable_support_and_keeps_static_asset(self):
        result,_=self.apply();self.assertEqual(result.layers['occupancy'][20,20],0)
        self.local.layers['occupancy'][:]=-1
        later,_=self.apply();self.assertEqual(later.layers['occupancy'][20,20],-1)
        self.assertEqual(self.prior.layers['occupancy'][20,20],-1)

    def test_outside_local_coverage_and_large_tilt_cannot_supply_free(self):
        transform=np.eye(3);transform[0,2]=4.
        result,_=self.apply(transform=transform)
        self.assertEqual(result.layers['occupancy'][20,20],-1)
        c,s=np.cos(.2),np.sin(.2);tilt=np.eye(4);tilt[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
        self.scene.record['prior']['map_from_local']=tilt.tolist()
        result,report=self.apply();self.assertEqual(result.layers['occupancy'][20,20],-1)
        self.assertEqual(report['reason'],'map_local_tilt_exceeds_planar_limit')

    def test_vector_projection_matches_independent_polygon_intersections(self):
        rng=np.random.default_rng(1105)
        cells=rng.choice([0,-1,100],size=(60,60),p=[.85,.08,.07]).astype(np.int8)
        self.local.layers['occupancy'][:]=cells
        y,x=np.indices(cells.shape);centres=np.c_[x.ravel()+.5,y.ravel()+.5]*.1
        points=rng.uniform(1,5,(20,2));corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*.07
        for angle in (.17,.8,1.4):
            c,s=np.cos(angle),np.sin(angle);rotation=np.array([[c,-s],[s,c]])
            free,hit=overlap_states(points,rotation,.14,self.local,cells)
            for i,p in enumerate(points):
                keep=polygon_cells(corners@rotation+p,centres,.05)
                values=cells.ravel()[keep]
                self.assertEqual(free[i],bool(np.all(values==0)))
                self.assertEqual(hit[i],bool(np.any(values==100)))

    def test_mapped_search_and_current_check_accept_the_same_known_free_start(self):
        cfg=load_config();geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        source=asset();source.snapshot.layers['occupancy'][20,20]=-1
        planner=Planner(cfg,geometry)
        result=planner.query(request('mapped',goal=[4.,2.]),scene=self.scene,asset=source,now=10.,clock_domain='synthetic')
        self.assertEqual(result['status'],'path_found',result)
        self.assertEqual(result['local_start_footprint'],'free')
        self.assertEqual(result['reference_overlay']['unknown_to_free'],1)
        self.assertEqual(source.snapshot.layers['occupancy'][20,20],-1)
        self.assertTrue(result['current_path_valid'])


if __name__=='__main__':unittest.main()
