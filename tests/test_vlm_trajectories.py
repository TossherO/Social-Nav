"""Physical invariants for causal VLM ego/person trajectory rendering."""
import unittest
import io
import numpy as np
from PIL import Image

from decision.vlm.trajectory_inputs import build_trajectories
from decision.vlm.scene_inputs import (place_bev_labels,build_window,bev_xy,clip_segment,
    render_observation,pointcloud_background,MODEL_SIZE,RGB_RECT,BEV_RECT,rgb_geometry)


def entry(index, x=0., yaw=0., state='observed'):
    c,s=np.cos(yaw),np.sin(yaw)
    matrix=np.array([[c,-s,0,x],[s,c,0,0],[0,0,1,0],[0,0,0,1]],dtype=float)
    stamp=1000+index*.1
    row=dict(stamp_s=stamp,pose_stamp_s=stamp,health='valid',pose_health='valid',pose_source='localization',
             world_from_lidar=matrix.tolist(),sequence_id='bag',session_id='people',generation=0,
             clock_domain='sim',frame_id='odom',detection_frame_id='rslidar',input_fingerprint=str(index),
             localization_binding=dict(session_id='local',generation=0,frame_id='odom',source='roll_fast_lio'),
             tracks=[dict(track_id='person',lifecycle_generation=0,confirmed=True,state=state,
                          box=[8.,2.,0.,.7,.7,1.7,0.,.9])])
    return dict(id=str(index),stamp_ns=round(stamp*1e9),result=row)


class TrajectoryTests(unittest.TestCase):
    def trace(self, rows, reference='2'):
        return build_trajectories(rows,[reference],{'person/g0':0})[reference]

    def test_static_world_person_stays_static_after_ego_translation_and_rotation(self):
        rows=[entry(0,0),entry(1,1),entry(2,2,np.pi/2)]
        trace=self.trace(rows)
        self.assertEqual(trace['status'],'available')
        np.testing.assert_allclose(np.asarray(trace['ego'][0])[:,1:],[ [0,2],[0,1],[0,0]],atol=1e-4)
        points=trace['people']['person/g0'][0]['points']
        np.testing.assert_allclose(np.asarray(points)[:,1:3],[[2,-6]]*3,atol=1e-4)

    def test_future_observation_cannot_change_earlier_image(self):
        rows=[entry(0),entry(1),entry(2),entry(3,100)]
        expected=self.trace(rows)
        rows[-1]['result']['tracks'][0]['box'][0]=999
        self.assertEqual(self.trace(rows),expected)
        first=self.trace(rows,'0')
        self.assertEqual(sum(len(s) for s in first['ego']),1)

    def test_prediction_edges_are_dashed_and_measurement_flags_remain_explicit(self):
        rows=[entry(0),entry(1,state='predicted'),entry(2),entry(3)]
        segments=self.trace(rows,'3')['people']['person/g0']
        self.assertEqual([s['observed'] for s in segments],[True,False,True])
        self.assertEqual([p[3] for p in segments[1]['points']],[True,False,True])

    def test_reset_and_invalid_pose_break_history(self):
        rows=[entry(0),entry(1),entry(2)]
        rows[1]['result']['health']='invalid'
        rows[2]['result']['localization_binding']['generation']=1
        trace=self.trace(rows)
        self.assertEqual(sum(len(s) for s in trace['ego']),1)
        self.assertEqual(sum(len(s['points']) for s in trace['people']['person/g0']),1)

    def test_missing_future_stale_or_raw_odometry_pose_is_not_fabricated(self):
        for field,value in [('pose_stamp_s',None),('pose_stamp_s',1001),('pose_stamp_s',999),('pose_source','raw_odometry')]:
            rows=[entry(2)];rows[0]['result'][field]=value
            trace=self.trace(rows)
            self.assertEqual(trace['status'],'unavailable')
            self.assertEqual(trace['ego'],[])

    def test_unobserved_gaps_and_history_budget_are_not_bridged(self):
        rows=[entry(0),entry(1),entry(8),entry(40)]
        trace=self.trace(rows,'8')
        self.assertEqual([len(segment) for segment in trace['ego']],[2,1])
        last=self.trace(rows,'40')
        self.assertEqual(sum(len(segment) for segment in last['ego']),1)


class BevLabelTests(unittest.TestCase):
    def test_dense_ids_keep_every_identity_without_covering_labels_or_markers(self):
        markers=[(i,1040+(i%2)*4,280+(i%3)*3) for i in range(20)]
        sizes={i:(34,23) for i in range(20)}
        labels=place_bev_labels(markers,sizes,(802,58,1278,534))
        self.assertEqual(labels,place_bev_labels(markers,sizes,(802,58,1278,534)))
        self.assertEqual([key for key,_ in labels],list(range(20)))
        def overlap(a,b):
            return min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])
        for i,(_,rect) in enumerate(labels):
            self.assertTrue(802<=rect[0]<rect[2]<=1278 and 58<=rect[1]<rect[3]<=534)
            self.assertFalse(any(overlap(rect,other) for _,other in labels[i+1:]))
            self.assertFalse(any(overlap(rect,(x-7,y-7,x+7,y+7)) for _,x,y in markers))

    def test_edge_labels_remain_inside_the_bev(self):
        markers=[(0,803,59),(1,1277,59),(2,803,533),(3,1277,533)]
        for _,rect in place_bev_labels(markers,{i:(40,23) for i in range(4)},(802,58,1278,534)):
            self.assertTrue(802<=rect[0]<rect[2]<=1278 and 58<=rect[1]<rect[3]<=534)


class CompactInputTests(unittest.TestCase):
    extent=(-4.8,14.4,-9.6,9.6)

    def frame(self):
        return dict(id='f',stamp_ns=1_000_000_000,fingerprint='frame',has_rgb=True,image_size=[640,480],
            people=[dict(key='p',eligible=True,box=[8,2,0,.7,.7,1.7,0,.9],rgb_box=[100,80,180,360])])

    def test_equal_metric_scale_and_fixed_roi_keep_outside_id_without_expanding_view(self):
        current=self.frame();current['people'][0]['box'][0]=100
        window=build_window([current],current,'asset',extent=self.extent)
        self.assertEqual(window['fields']['e'],list(self.extent))
        self.assertEqual(window['fields']['v'],[0])
        self.assertEqual(window['fields']['o'][0][1][-1],[100,2])
        origin=np.array(bev_xy(0,0,self.extent))
        self.assertAlmostEqual(np.linalg.norm(np.array(bev_xy(1,0,self.extent))-origin),
                               np.linalg.norm(np.array(bev_xy(0,1,self.extent))-origin))

    def test_clipping_keeps_crossing_edges_and_rejects_outside_edges(self):
        np.testing.assert_allclose(clip_segment((-100,0),(100,0),(-5,15,-10,10)),((-5.,0.),(15.,0.)),atol=1e-12)
        self.assertIsNone(clip_segment((-100,50),(100,50),self.extent))
        self.assertIsNone(clip_segment((30,0),(30,5),self.extent))

    def test_compact_rgb_avoids_wide_letterboxing_and_no_background_grid(self):
        frame=self.frame();stream=io.BytesIO();Image.new('RGB',(640,480),'#809cad').save(stream,format='PNG')
        image=render_observation(frame,stream.getvalue(),{'p':0},0,self.extent)
        self.assertEqual(image.size,MODEL_SIZE)
        _,ox,oy,w,h=rgb_geometry(frame['image_size'])
        self.assertLess(ox-RGB_RECT[0],10)
        self.assertEqual(oy,RGB_RECT[1])
        self.assertEqual(h,RGB_RECT[3]-RGB_RECT[1])
        self.assertGreater(w,700)
        colors=np.asarray(image.crop(BEV_RECT))
        self.assertFalse(np.any(np.all(colors==[212,220,229],axis=-1)))

    def test_full_height_background_preserves_out_of_color_range_points_and_is_order_invariant(self):
        points=np.array([[2,1,-100],[3,2,100],[3,2,0],[np.nan,0,0]],np.float32)
        a=pointcloud_background(points,self.extent);b=pointcloud_background(points[::-1],self.extent)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.size,(BEV_RECT[2]-BEV_RECT[0],BEV_RECT[3]-BEV_RECT[1]))
        self.assertGreater(np.count_nonzero(np.any(np.asarray(a)!=[245,247,250],axis=-1)),2)

    def test_boundary_markers_and_thick_trajectories_do_not_spill_into_rgb_or_header(self):
        frame=self.frame();frame['people'][0]['box'][:2]=[14.4,9.6];frame['people'][0]['rgb_box']=None
        data=io.BytesIO();Image.new('RGB',(640,480),'#809cad').save(data,format='PNG')
        base=render_observation(frame,data.getvalue(),{},0,self.extent)
        image=render_observation(frame,data.getvalue(),{'p':0},0,self.extent)
        np.testing.assert_array_equal(np.asarray(base)[:,:RGB_RECT[2]],np.asarray(image)[:,:RGB_RECT[2]])
        np.testing.assert_array_equal(np.asarray(base)[:32],np.asarray(image)[:32])

    def test_height_scale_is_explicit_and_trajectory_is_a_plain_foreground_line(self):
        from PIL import ImageColor
        from decision.vlm.scene_inputs import BEV_HISTORY
        frame=self.frame();data=io.BytesIO()
        Image.new('RGB',(640,480),'#809cad').save(data,format='PNG')
        cloud=pointcloud_background(np.array([[5,2,1.]],np.float32),self.extent)
        trace=dict(status='available',ego=[],people={'p':[dict(observed=True,
            points=[[-1.,2.,2.,True],[-.5,5.,2.,True],[0.,8.,2.,True]])]})
        image=render_observation(frame,data.getvalue(),{'p':0},0,self.extent,trace,bev_background=cloud)
        plain=cloud.copy();plain.info.clear()
        without=render_observation(frame,data.getvalue(),{'p':0},0,self.extent,trace,bev_background=plain)
        self.assertFalse(np.array_equal(np.asarray(image)[:32,260:736],np.asarray(without)[:32,260:736]))
        np.testing.assert_array_equal(np.asarray(image)[32:,:736],np.asarray(without)[32:,:736])
        x,y=map(int,bev_xy(5,2,self.extent));blue=ImageColor.getrgb(BEV_HISTORY)
        self.assertEqual(image.getpixel((x,y)),blue)
        from decision.vlm.scene_inputs import TRAJECTORY_WIDTH
        core=sum(image.getpixel((px,y))==blue for px in range(x-5,x+6))
        self.assertGreater(core,0)
        self.assertLessEqual(core,TRAJECTORY_WIDTH)

    def test_prediction_style_changes_merge_but_real_gaps_remain_and_flags_are_preserved(self):
        import copy
        from decision.vlm.scene_inputs import continuous_history
        segments=[dict(observed=True,points=[[-1.,2.,2.,True],[-.8,3.,2.,True]]),
                  dict(observed=False,points=[[-.8,3.,2.,True],[-.6,4.,2.,False],[-.4,5.,2.,True]]),
                  dict(observed=True,points=[[-.4,5.,2.,True],[0.,6.,2.,True]]),
                  dict(observed=True,points=[[1.,6.,2.,True],[1.1,7.,2.,True]])]
        before=copy.deepcopy(segments);runs=list(continuous_history(segments))
        self.assertEqual([len(r) for r in runs],[5,2])
        self.assertFalse(runs[0][2][3])
        self.assertEqual(segments,before)


if __name__=='__main__':unittest.main()
