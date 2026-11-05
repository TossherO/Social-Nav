"""Zero person lifetime keeps current hits and withdraws only earlier evidence."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from perception.environment.layers import GridSnapshot,load_config
from perception.environment.occupancy import OccupancyTree
from perception.environment.near_field import NearFieldFreeMemory
from tests.test_environment_near_field import sample


def voxel_sequence(backend,ttl,mixed=False):
    config=load_config();config['dynamic']['transient_ttl_s']=ttl
    tree=OccupancyTree(.1,backend=backend)
    query=np.array([[.55,.05,.5],[1.05,.05,.5]])
    bounds=[-4,-4,-2,4,4,3];points=query.copy();mask=[1,0]
    if mixed:points=np.r_[points,[[.56,.05,.5]]];mask=[1,0,0]
    states=[]
    try:
        for stamp,cloud,moving in ((1.,points,mask),(1.1,[],[]),(1.3,[],[]),(1.5,[],[]),(1.7,[],[]),
                                    (1.8,points,mask),(1.9,points,mask)):
            tree.update(np.asarray(cloud).reshape(-1,3),[0,0,.84],bounds,stamp,config,rays=False,
                transient_mask=moving,transient_ttl=ttl,read=False)
            values=tree.snapshot(stamp)
            keys=np.floor(query/.1).astype(int)
            occupied=np.floor(values[values[:,3]>=config['occupied_probability'],:3]/.1).astype(int)
            states.append([bool((occupied==key).all(axis=1).any()) for key in keys])
        return states
    finally:tree.close()


def overlay_sequence(backend,known):
    from perception.environment.cuda_view import native_view
    config=load_config();config['dynamic']['transient_ttl_s']=0.
    memory=NearFieldFreeMemory(config,dict(remember_outside_ring=True,temporary_obstacle_overlay=True))
    tree=OccupancyTree(.1,backend=backend);states=[];times=[]
    try:
        for stamp,source,points,moving in ((1.,sample(1.,state=0 if known else -1),[],[]),
                (2.,sample(2.),[[.55,.05,.5]],[1]),(2.2,sample(2.2),[],[]),
                (2.4,sample(2.4),[[.55,.05,.5]],[1])):
            # Unknown occupancy can still have measured ground without free rays.
            if not known:source.layers['ground_support'][40,45]=True
            cloud=np.asarray(points).reshape(-1,3)
            if backend=='cpu':view=memory.update(source,cloud,transient_mask=moving).layers
            else:
                raw=source.layers
                view,_=native_view(tree,raw['ground_z'],raw['ground_support'],[-4,-4],config,stamp,cloud,
                    source_layers=raw,metadata=source.metadata,policy=memory,transient_mask=moving)
            states.append(int(view['occupancy'][40,45]));times.append(float(view['last_observed'][40,45]))
        return states,times
    finally:tree.close()


class PersonLifetimeTests(unittest.TestCase):
    def test_zero_config_is_explicit_and_negative_nonfinite_are_rejected(self):
        config=load_config();self.assertEqual(config['dynamic']['transient_ttl_s'],.2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'environment.yaml'
            for ttl in (0.,.2,-.1,np.nan,np.inf):
                config['dynamic']['transient_ttl_s']=ttl;path.write_text(yaml.safe_dump(config))
                if np.isfinite(ttl) and ttl>=0:self.assertEqual(load_config(path)['dynamic']['transient_ttl_s'],ttl)
                else:
                    with self.assertRaises(ValueError):load_config(path)

    def test_zero_keeps_current_and_repeated_hits_but_not_an_unobserved_person_voxel(self):
        states=voxel_sequence('cpu',0.)
        self.assertEqual([s[0] for s in states],[True,False,False,False,False,True,True])
        self.assertTrue(all(s[1] for s in states))

    def test_positive_lifetimes_retain_short_gaps_and_current_mixed_structure_is_ordinary(self):
        for ttl,expected in ((.2,[True,True,False,False,False,True,True]),
                             (.4,[True,True,True,False,False,True,True]),
                             (.6,[True,True,True,True,False,True,True])):
            with self.subTest(ttl=ttl):self.assertEqual([s[0] for s in voxel_sequence('cpu',ttl)],expected)
        self.assertTrue(all(s[0] for s in voxel_sequence('cpu',0.,mixed=True)))

    def test_zero_overlay_restores_only_existing_background_and_its_original_time(self):
        known,times=overlay_sequence('cpu',True)
        self.assertEqual(known,[0,100,0,100]);self.assertEqual(times[2],1.)
        self.assertEqual(overlay_sequence('cpu',False)[0],[-1,100,-1,100])

    def test_negative_native_wrapper_lifetime_is_rejected(self):
        tree=OccupancyTree(.1);self.addCleanup(tree.close)
        with self.assertRaises(ValueError):tree.update(np.array([[.55,.05,.5]]),[0,0,.84],[-4,-4,-2,4,4,3],
            1.,load_config(),transient_mask=[1],transient_ttl=-.1)


class PersonLifetimeCudaTests(unittest.TestCase):
    def test_zero_and_short_lifetimes_and_near_field_match_cpu(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        for ttl in (0.,.2,.4,.6):
            for mixed in (False,True):self.assertEqual(voxel_sequence('cpu',ttl,mixed),voxel_sequence('cuda',ttl,mixed))
        for known in (False,True):
            a,b=overlay_sequence('cpu',known),overlay_sequence('cuda',known)
            self.assertEqual(a[0],b[0]);np.testing.assert_equal(a[1],b[1])
