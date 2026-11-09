"""Near-field history is distinct from fresh measurement and unobserved space."""
import unittest
import numpy as np
from perception.environment.layers import GridSnapshot, load_config, finish_evidence
from perception.environment.near_field import NearFieldFreeMemory, ground_blind_zone
from perception.environment.occupancy import OccupancyTree


def sample(stamp, *, state=-1, identity=0, sensor_x=0., origin_x=-4., floor=0.):
    shape=(80,80);z=np.full(shape,floor,np.float32);support=np.zeros(shape,bool)
    occupancy=np.full(shape,-1,np.int8);seen=np.full(shape,-np.inf);coverage=np.zeros(shape,np.float32)
    x=int(np.floor((.55-origin_x)/.1));y=40
    if 0<=x<80:
        occupancy[y,x]=state;support[y,x]=state==0;coverage[y,x]=float(state==0);seen[y,x]=stamp if state==0 else -np.inf
    layers=finish_evidence(occupancy,np.isfinite(seen),seen,coverage,np.full(shape,np.nan),
                           np.full(shape,np.nan),z,support,.1)
    transform=np.eye(4);transform[:3,3]=[sensor_x,0,.84]
    return GridSnapshot(dict(stamp_s=stamp,snapshot_id='test:%s:%s'%(identity,stamp),session_id='test',
        generation=identity,frame_id='odom',resolution=.1,shape=list(shape),origin_xy=[origin_x,-4.],
        world_from_lidar=transform.tolist(),ground_tracking=dict(plane_world=[0,0,0],reference_height_m=.84)),layers)


def column_sequence(backend):
    cfg=load_config();cfg.update(hit_invalidates_free=True,voxel_ttl_s=120.)
    tree=OccupancyTree(.1,backend=backend);states=[];bounds=[-3,-3,-3,3,3,3]
    z=.15+.1*np.arange(17);far=np.c_[np.full(17,2.05),np.full(17,.05),z]
    origins=np.c_[np.full(17,.05),np.full(17,.05),z];near=np.array([[1.05,.05,.55]])
    floor=np.zeros((60,60));support=np.ones_like(floor,bool)
    def read(stamp,current):
        p=tree.project(floor,support,[-3,-3],cfg,stamp,endpoints=current)
        states.append(int(p['occupancy'][30,40]))
    try:
        for s in np.arange(8)*.2+1:tree.update(far,[.05,.05,.05],bounds,float(s),cfg,ray_origins=origins)
        read(2.4,np.empty((0,3)))
        tree.update(near,[.05,.05,.05],bounds,3.,cfg,rays=False);read(3.,near)
        tree.update(np.empty((0,3)),[.05,.05,.05],bounds,3.2,cfg,rays=False);read(3.2,np.empty((0,3)))
        tree.update(np.empty((0,3)),[.05,.05,.05],bounds,6.,cfg,rays=False);read(6.,np.empty((0,3)))
        for s in [6.2,6.4,6.6,6.8]:tree.update(far,[.05,.05,.05],bounds,s,cfg,ray_origins=origins)
        read(6.8,np.empty((0,3)))
        return states
    finally:tree.close()


def person_background_sequence(backend, *, suspended=False):
    """Real resident evidence: person-only, unknown background, then mixed chair/person."""
    from perception.environment.cuda_view import native_view
    config=dict(load_config(),resolution=.1,hit_invalidates_free=True)
    memory=NearFieldFreeMemory(config,dict(remember_outside_ring=True,temporary_obstacle_overlay=True))
    tree=OccupancyTree(.1,backend=backend);seed=sample(1,state=0);shape=(80,80)
    try:
        if backend=='cpu':memory.update(seed,np.empty((0,3)))
        else:native_view(tree,seed.layers['ground_z'],seed.layers['ground_support'],[-4,-4],config,1.,np.empty((0,3)),
                source_layers=seed.layers,metadata=seed.metadata,policy=memory)
        results=[]
        for stamp,points,moving in [(2.,[[.55,.05,.5],[1.05,.05,.5]],[1,1]),(2.1,[[.55,.05,1.]],[0]),
                (2.2,[[.55,.05,1.]],[1])]:
            points=np.asarray(points);moving=np.asarray(moving,bool)
            tree.update(points,[0,0,.84],[-4,-4,-2,4,4,3],stamp,config,rays=False,
                transient_mask=moving,transient_ttl=.8,read=False)
            metadata=sample(stamp).metadata
            metadata['ground_tracking']['history_suspended']=suspended
            layers=tree.project(np.zeros(shape,np.float32),np.ones(shape,bool),[-4,-4],config,stamp,endpoints=points,
                view_metadata=metadata,near_field=memory if backend=='cuda' else None,transient_mask=moving)
            if backend=='cpu':layers=memory.update(GridSnapshot(metadata,layers),points,transient_mask=moving,person_only=tree.person_only).layers
            results.append(layers)
        return results
    finally:tree.close()


def suspended_history_sequence(backend):
    """A false ground alarm retains provenance while new obstacles still arrive."""
    from perception.environment.cuda_view import native_view
    config=load_config();memory=NearFieldFreeMemory(config,{k:v for k,v in config['near_field'].items() if k!='enabled'})
    tree=OccupancyTree(.1,backend=backend);results=[]
    try:
        for stamp,state,paused,points in [(1.,0,False,[]),(2.,0,True,[]),
                (3.,-1,True,[[.55,.05,.5]]),(3.1,-1,True,[]),(3.2,-1,False,[]),(6.,-1,False,[])]:
            seed=sample(stamp,state=state);seed.metadata['ground_tracking']['history_suspended']=paused
            if stamp==2.:
                # A new raw-free column during suspension must not enter memory.
                for key,value in [('occupancy',0),('ground_support',True),('coverage',1.),('last_observed',stamp)]:
                    seed.layers[key][40,46]=value
            points=np.asarray(points,dtype=float).reshape(-1,3)
            if backend=='cpu':
                view=memory.update(seed,points);results.append((view.layers,view.metadata['near_field_memory']))
            else:
                results.append(native_view(tree,seed.layers['ground_z'],seed.layers['ground_support'],[-4,-4],config,stamp,points,
                    source_layers=seed.layers,metadata=seed.metadata,policy=memory))
        return results
    finally:tree.close()


class NearFieldTests(unittest.TestCase):
    def model(self):
        return NearFieldFreeMemory(dict(load_config(),hit_invalidates_free=True))

    def test_local_ground_conflict_blocks_only_its_cached_column_and_preserves_obstacles(self):
        config=load_config();memory=NearFieldFreeMemory(config,{k:v for k,v in config['near_field'].items() if k!='enabled'})
        seed=sample(1.,state=0)
        for name,value in (('occupancy',0),('ground_support',True),('coverage',1.),('last_observed',1.)):
            seed.layers[name][40,55]=value
        memory.update(seed,np.empty((0,3)))
        alarm=sample(2.);alarm.layers['ground_conflict']=np.zeros((80,80),bool)
        alarm.layers['ground_conflict'][40,45]=True
        result=memory.update(alarm,np.empty((0,3)))
        self.assertEqual(result.layers['occupancy'][40,45],-1)
        self.assertEqual(result.layers['occupancy'][40,55],0)
        self.assertTrue(np.isfinite(memory.confirmed[40,45]))
        recovered=memory.update(sample(2.2),np.empty((0,3)))
        self.assertEqual(recovered.layers['occupancy'][40,45],0)
        # The conflict field is supplied by the causal ground model. This memory
        # neither expires nor clears it; it continues classifying new obstacles.
        alarm=sample(2.4);alarm.layers['ground_conflict']=np.zeros((80,80),bool)
        alarm.layers['ground_conflict'][40,45]=True
        occupied=memory.update(alarm,np.array([[.55,.05,.5]]))
        self.assertEqual(occupied.layers['occupancy'][40,45],100)

    def test_cuda_local_ground_conflicts_match_cpu_cached_free_and_person_background(self):
        import torch
        from perception.environment.cuda_view import native_view
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        config=load_config();tree=OccupancyTree(.1,backend='cuda');self.addCleanup(tree.close)
        cpu=self.model();gpu=self.model()
        for i,paused in enumerate((False,True,True,False)):
            raw=sample(1.+i*.2,state=0 if i==0 else -1)
            raw.layers['ground_conflict']=np.zeros((80,80),bool)
            raw.layers['ground_conflict'][40,45]=paused
            expected=cpu.update(raw,np.empty((0,3)))
            actual,_=native_view(tree,raw.layers['ground_z'],raw.layers['ground_support'],[-4,-4],config,
                raw.metadata['stamp_s'],np.empty((0,3)),source_layers=raw.layers,metadata=raw.metadata,policy=gpu)
            for name in ('occupancy','ground_support','ground_conflict','near_field_free','last_observed'):
                np.testing.assert_equal(actual[name],expected.layers[name],err_msg=name)

    def test_ground_suspension_retains_times_and_obstacles_without_new_free(self):
        views=suspended_history_sequence('cpu')
        self.assertEqual([int(v['occupancy'][40,45]) for v,_ in views],[0,-1,100,100,100,0])
        for layers,description in views[1:4]:
            self.assertFalse(layers['near_field_free'].any())
            self.assertTrue(description['history_suspended'])
            self.assertEqual(description['cached_cells'],1)
        self.assertEqual(views[-1][0]['near_field_free_classified_at'][40,45],1.)
        self.assertEqual(views[-1][0]['last_observed'][40,45],1.)
        self.assertEqual(views[-1][0]['occupancy'][40,46],-1)

    def test_suspended_person_columns_cannot_restore_historical_background(self):
        for layers in person_background_sequence('cpu',suspended=True):
            self.assertFalse(layers['person_background_free'].any())

    def test_person_background_requires_free_history_and_excludes_mixed_ordinary_evidence(self):
        first,mixed,hidden=person_background_sequence('cpu')
        self.assertEqual(first['occupancy'][40,45],100)
        self.assertTrue(first['person_background_free'][40,45])
        self.assertFalse(first['person_background_free'][40,50])
        self.assertFalse(mixed['person_background_free'][40,45])
        self.assertEqual(mixed['occupancy'][40,45],100)
        self.assertFalse(hidden['person_background_free'][40,45])

    def test_previous_free_survives_long_blind_stop_without_claiming_fresh_rays(self):
        memory=self.model();memory.update(sample(1,state=0),np.empty((0,3)))
        result=memory.update(sample(1000),np.empty((0,3)))
        self.assertEqual(result.layers['occupancy'][40,45],0)
        self.assertEqual(result.layers['last_observed'][40,45],1.)
        self.assertEqual(result.layers['near_field_free_classified_at'][40,45],1.)
        self.assertEqual(int((result.layers['occupancy']==0).sum()),1)
        self.assertEqual(result.metadata['near_field_memory']['restored_cells'],1)
        self.assertFalse(result.layers['occupancy'].flags.writeable)

    def test_current_hit_uses_cached_ground_and_history_recovers_after_expiry(self):
        memory=self.model();memory.update(sample(1,state=0),np.empty((0,3)))
        hit=memory.update(sample(2),np.array([[.55,.05,.5]]))
        self.assertEqual(hit.metadata['near_field_memory']['invalidated_hit_cells'],0)
        self.assertEqual(hit.layers['occupancy'][40,45],100)
        later=memory.update(sample(20),np.empty((0,3)))
        self.assertEqual(later.layers['occupancy'][40,45],0)
        memory.update(sample(21,state=0),np.empty((0,3)))
        self.assertEqual(memory.update(sample(100),np.empty((0,3))).layers['occupancy'][40,45],0)

    def test_leaving_blind_zone_drops_memory_and_return_does_not_restore_it(self):
        memory=self.model();memory.update(sample(1,state=0),np.empty((0,3)))
        result=memory.update(sample(2,sensor_x=5),np.empty((0,3)))
        self.assertEqual(result.layers['occupancy'][40,45],-1)
        self.assertEqual(memory.update(sample(3),np.empty((0,3))).layers['occupancy'][40,45],-1)

    def test_grid_shift_lifecycle_change_and_backward_seek(self):
        memory=self.model();memory.update(sample(1,state=0),np.empty((0,3)))
        moved=memory.update(sample(2,origin_x=-3.9),np.empty((0,3)))
        self.assertEqual(moved.layers['occupancy'][40,44],0)
        reset=memory.update(sample(3,identity=1),np.empty((0,3)))
        self.assertFalse((reset.layers['occupancy']==0).any())
        memory.update(sample(4,state=0,identity=1),np.empty((0,3)))
        back=memory.update(sample(2,identity=1),np.empty((0,3)))
        self.assertFalse((back.layers['occupancy']==0).any())

    def test_ground_change_drops_old_free_and_current_occupied_is_never_overwritten(self):
        memory=self.model();memory.update(sample(1,state=0),np.empty((0,3)))
        changed=sample(2,state=0,floor=.4)
        changed.layers['occupancy'][40,45]=-1
        result=memory.update(changed,np.empty((0,3)))
        self.assertEqual(result.layers['occupancy'][40,45],-1)
        self.assertEqual(result.metadata['near_field_memory']['ground_conflict_cells'],1)
        memory.update(sample(3,state=0),np.empty((0,3)))
        occupied=memory.update(sample(4,state=100),np.empty((0,3)))
        self.assertEqual(occupied.layers['occupancy'][40,45],100)

    def test_ring_respects_slope_and_missing_geometry(self):
        memory=self.model();s=sample(1)
        flat,_=ground_blind_zone(s,np.array([0,0,0]),memory.settings)
        slope,_=ground_blind_zone(s,np.array([.1,0,0]),memory.settings)
        self.assertGreater(int(flat[:,40:].sum()),int(slope[:,40:].sum()))
        self.assertLess(int(flat[:,:40].sum()),int(slope[:,:40].sum()))
        absent,_=ground_blind_zone(s,None,memory.settings)
        self.assertFalse(absent.any())

    def test_window_candidates_are_only_published_in_blind_zone(self):
        memory=NearFieldFreeMemory(dict(load_config(),hit_invalidates_free=True),dict(remember_outside_ring=True))
        memory.update(sample(1,state=0),np.empty((0,3)))
        outside=memory.update(sample(2,sensor_x=5),np.empty((0,3)))
        self.assertEqual(outside.layers['occupancy'][40,45],-1)
        returned=memory.update(sample(100,sensor_x=0),np.empty((0,3)))
        self.assertEqual(returned.layers['occupancy'][40,45],0)

    def test_expired_hit_outside_blind_zone_preserves_window_candidate(self):
        memory=NearFieldFreeMemory(dict(load_config(),hit_invalidates_free=True),dict(remember_outside_ring=True))
        memory.update(sample(1,state=0),np.empty((0,3)))
        memory.update(sample(2,sensor_x=5),np.array([[.55,.05,.5]]))
        returned=memory.update(sample(100,sensor_x=0),np.empty((0,3)))
        self.assertEqual(returned.layers['occupancy'][40,45],0)

    def test_retired_immediate_invalidation_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'only supports temporary obstacle overlay'):
            NearFieldFreeMemory(load_config(),dict(temporary_obstacle_overlay=False))

    def test_background_free_survives_temporary_hits_but_does_not_hide_active_obstacle(self):
        memory=NearFieldFreeMemory(dict(load_config(),hit_invalidates_free=True),
            dict(remember_outside_ring=True,temporary_obstacle_overlay=True))
        memory.update(sample(1,state=0),np.empty((0,3)))
        occupied=memory.update(sample(2),np.array([[.55,.05,.5]]))
        self.assertEqual(occupied.layers['occupancy'][40,45],100)
        self.assertEqual(occupied.layers['last_observed'][40,45],2.)
        held=memory.update(sample(3),np.empty((0,3)))
        self.assertEqual(held.layers['occupancy'][40,45],100)
        restored=memory.update(sample(5),np.empty((0,3)))
        self.assertEqual(restored.layers['occupancy'][40,45],0)
        self.assertEqual(restored.layers['last_observed'][40,45],1.)
        self.assertEqual(restored.layers['near_field_obstacle_stamp'][40,45],2.)
        # Continuous observations extend the temporary obstacle, not its background.
        for t in [6,7,8]:
            result=memory.update(sample(t),np.array([[.55,.05,.5]]))
            self.assertEqual(result.layers['occupancy'][40,45],100)

    def test_ground_history_can_support_new_obstacle_without_previous_free_column(self):
        memory=NearFieldFreeMemory(dict(load_config(),hit_invalidates_free=True),
            dict(remember_outside_ring=True,temporary_obstacle_overlay=True))
        initial=sample(1,state=100);initial.layers['ground_support'][40,45]=True
        memory.update(initial,np.empty((0,3)))
        occupied=memory.update(sample(2),np.array([[.55,.05,.5]]))
        self.assertEqual(occupied.layers['occupancy'][40,45],100)
        self.assertEqual(occupied.layers['obstacle_min_z'][40,45],.5)
        self.assertEqual(occupied.layers['near_field_obstacle_stamp'][40,45],2.)
        expired=memory.update(sample(5),np.empty((0,3)))
        self.assertEqual(expired.layers['occupancy'][40,45],-1)

    def test_person_coverage_uses_shorter_expiry_and_mixed_hits_keep_ordinary_expiry(self):
        for moving,expected in [([1,1],2.+load_config()['dynamic']['transient_ttl_s']),([1,0],4.)]:
            memory=NearFieldFreeMemory(dict(load_config(),hit_invalidates_free=True),
                dict(remember_outside_ring=True,temporary_obstacle_overlay=True))
            memory.update(sample(1,state=0),np.empty((0,3)))
            result=memory.update(sample(2),np.array([[.55,.05,.4],[.55,.05,.5]]),transient_mask=moving)
            self.assertAlmostEqual(result.layers['near_field_obstacle_expires_at'][40,45],expected)
            later=memory.update(sample(3),np.empty((0,3)))
            self.assertEqual(later.layers['occupancy'][40,45],0 if expected<3 else 100)

    def test_new_hit_removes_old_free_until_new_free_rays_confirm_it(self):
        self.assertEqual(column_sequence('cpu'),[0,100,100,-1,0])


if __name__=='__main__':unittest.main()
