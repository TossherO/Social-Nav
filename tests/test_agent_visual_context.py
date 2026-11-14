"""Static map prefixes, causal trajectories and explicit saved visual memory."""
import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import httpx
import numpy as np
from PIL import Image, ImageDraw
from decision.agent import CONFIG_DIR
from decision.agent.core.model import Model, cache_messages, input_bound
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Budget, Store
from decision.agent.core.trajectory import Trajectory
from decision.agent.tools.places import Catalog
from decision.agent.tools.spatial import ReferenceMap, SpatialView, display_extent, map_font, wrap_map_text, reference_layout, render, semantic_label_layout
from perception.environment.layers import GridSnapshot
from tests.test_agent import ScriptedFeedback, reply
from tests.test_agent_conversation import wire
from tests import test_agent_goals


class VisualContextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = Store(self.root/'session.sqlite')
        self.budget = Budget(self.root/'budget.sqlite')
        self.nav = ScriptedFeedback()
        self.grid = GridSnapshot(dict(frame_id='sim_odom',shape=[100,140],resolution=.1,
            origin_xy=[0.,0.],snapshot_id='unit-static'),dict(occupancy=np.zeros((100,140),np.int8)))
        self.nav.reference_grid = lambda:self.grid
        self.nav.spatial_inputs = lambda:dict(global_grid=self.grid,scene=None,pose=self.nav.state['pose'],
            stamp_s=self.nav.state['stamp_s'],clock_domain=self.nav.clock_domain)
        self.nav.observe = lambda:test_agent_goals.IndoorGoalTests.rgb(self)
        self.session = AgentSession('先到电梯口再到大厅',Catalog.load(CONFIG_DIR/'simulation.json'),self.nav,self.store)
        config = self.root/'unit.json'
        config.write_text(json.dumps(dict(base_url='https://unit.invalid/v1',api_key='unit-fixture')))
        self.requests = []
        def handle(request):
            self.requests.append(json.loads(request.content))
            return httpx.Response(200,text=wire(reply('lookup_places',name='').message['tool_calls']))
        self.client = Model(config,self.budget,self.store,transport=httpx.MockTransport(handle))

    def tearDown(self):
        self.budget.close();self.store.close();self.temporary.cleanup()

    def request(self):
        payload = self.session.prompt()
        self.client.decide(payload,observation=self.session.observation,spatial_view=self.session.spatial_view)
        return self.requests[-1]

    def save(self):
        self.session.prompt()
        response = reply('save_keyframe',observation_id=self.session.observation.record['observation_id'],reason='重要入口')
        result = self.session.call('save_keyframe',_context=self.session.context(),**response.action['arguments'])
        self.assertEqual(result['status'],'saved')
        self.session.conversation.commit(response,result,keyframe=result['keyframe'])
        return result['keyframe']

    def test_static_prefix_contains_all_fixed_annotations_and_survives_task_and_pose_changes(self):
        first = self.request()
        reference = first['messages'][1]
        metadata = json.loads(reference['content'][0]['text'])['reference']
        self.assertEqual(max(metadata['image_size']),2048)
        self.assertGreater(metadata['image_size'][0],metadata['image_size'][1])
        self.assertTrue(metadata['annotations']['regions'])
        self.assertIn('front_window',[p['place_id'] for p in metadata['annotations']['points']])
        self.assertEqual([p['place_id'] for p in metadata['annotations']['points']],list(self.session.catalog.places))
        self.assertIsNone(next(p for p in metadata['annotations']['points'] if p['place_id']=='mission_start')['pose'])
        data = base64.b64decode(reference['content'][1]['image_url']['url'].split(',',1)[1])
        self.assertEqual(list(Image.open(io.BytesIO(data)).size),metadata['image_size'])
        self.nav.state.update(stamp_s=11.,pose=[4.5,4.,.2])
        self.session.event('replace','去另一个位置')
        second = self.request()
        self.assertEqual(reference,second['messages'][1])
        self.assertNotEqual(first['messages'][-1],second['messages'][-1])
        self.assertEqual(first['messages'][2]['content'],'先到电梯口再到大厅')

    def test_current_three_images_and_reference_map_are_separate_and_source_bound(self):
        body = self.request()
        self.assertEqual(sum(p['type']=='image_url' for p in body['messages'][-1]['content']),3)
        self.assertEqual(sum(p['type']=='image_url' for m in body['messages'] if isinstance(m.get('content'),list)
                             for p in m['content']),4)
        view = self.session.spatial_view.record
        self.assertEqual(view['image_roles'],['trajectory_global_overview','target_or_robot_static_crop'])
        self.assertEqual(view['source_stamp_s'],10.)
        self.assertEqual(view['trajectory']['sample_count'],1)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM events WHERE kind='reference_map_prepared'").fetchone()[0],1)

    def test_static_crop_keeps_camera_view_and_local_grid_binding(self):
        from decision.agent.tools.sensors import sensor_context
        from navigation.planning.footprint import VehicleGeometry
        geometry=VehicleGeometry([[-.3,-.3],[.5,-.3],[.5,.3],[-.3,.3]],np.eye(4),source={'vehicle_id':'wheelchair_tl_local'},assumed=False)
        info=dict(width=640,height=480,K=[600,0,320,0,600,240,0,0,1],frame_id='camera_color_optical_frame',stamp_ns=9_000_000_000)
        sensors=sensor_context(geometry,info,rgb_shape=[480,640,3],rgb_frame=info['frame_id'],rgb_stamp=9.9)
        local=GridSnapshot(dict(frame_id='odom',shape=[10,10],resolution=.1,origin_xy=[0.,0.],snapshot_id='local'),
                           dict(occupancy=np.zeros((10,10),np.int8)))
        inputs=dict(global_grid=self.grid,scene=None,local_grid=local,local_grid_observation_id='obs',
                    local_grid_map_from_local=np.eye(4).tolist(),geometry=geometry,sensors=sensors,
                    pose=[4.,4.,0.],stamp_s=10.,clock_domain='synthetic')
        view=SpatialView.build(self.session.catalog,inputs)
        self.assertEqual(view.record['local_source'],'static_prior_only')
        self.assertTrue(view.record['local_grid_present'])
        self.assertEqual(view.record['local_observation_id'],'obs')
        self.assertEqual(view.record['camera_view']['status'],'approximate_pinhole')
        self.assertIn('not_occlusion',view.record['camera_view']['interpretation'])
        pixels=np.asarray(Image.open(io.BytesIO(view.images[1])))
        self.assertTrue(np.any(np.all(pixels==[120,72,191],axis=2)))
        self.assertNotEqual(view.record['input_source_binding'],SpatialView.build(self.session.catalog,dict(inputs,sensors=None)).record['input_source_binding'])

    def test_reference_dimensions_adapt_without_distorting_the_metric_map(self):
        sizes = []
        for height,width in ((80,240),(140,140),(150,90)):
            with self.subTest(shape=(height,width)):
                grid = GridSnapshot(dict(self.grid.metadata,shape=[height,width]),
                                    dict(occupancy=np.zeros((height,width),np.int8)))
                reference = ReferenceMap.build(self.session.reference_catalog,grid)
                picture = Image.open(io.BytesIO(reference.image))
                sizes.append(picture.size)
                self.assertEqual(list(picture.size),reference.record['image_size'])
                self.assertLessEqual(max(picture.size),2048)
                pixels = np.asarray(picture)
                border = np.all(pixels==[82,98,116],axis=2)
                rows,columns = border.sum(axis=1),border.sum(axis=0)
                # Text antialiasing can share the border colour; identify the
                # continuous rectangle edges rather than isolated text pixels.
                yy = np.flatnonzero(rows>=.9*rows.max())
                xx = np.flatnonzero(columns>=.9*columns.max())
                self.assertAlmostEqual((xx.max()-xx.min())/(yy.max()-yy.min()),width/height,delta=.01)
                self.assertEqual(reference.record['annotations']['regions'],self.session.reference_catalog.annotations()['regions'])
                expected=[p for p in self.session.reference_catalog.annotations()['points'] if p['place_id']!='mission_start']
                actual=[p for p in reference.record['annotations']['points'] if p['place_id']!='mission_start']
                self.assertEqual(actual,expected)
        self.assertGreater(sizes[0][0],sizes[0][1])
        self.assertLess(sizes[2][0],sizes[2][1])
        self.assertEqual(len(set(sizes)),3)

    def test_display_crop_keeps_observed_cells_and_original_coordinates(self):
        cells=np.full((100,100),-1,np.int8);cells[30:50,40:60]=0;cells[20,70]=100
        grid=GridSnapshot(dict(self.grid.metadata,shape=[100,100]),dict(occupancy=cells))
        crop=display_extent(grid,dict(points=[],regions=[]))
        np.testing.assert_allclose(crop['bounds'],[3.,1.,8.1,6.])
        np.testing.assert_allclose(crop['asset_bounds'],[0.,0.,10.,10.])
        self.assertEqual(crop['policy'],'known_occupancy_with_padding')
        self.assertEqual(cells[0,0],-1);self.assertEqual(cells[20,70],100)
        empty=GridSnapshot(grid.metadata,dict(occupancy=np.full((100,100),-1,np.int8)))
        fallback=display_extent(empty,dict(points=[],regions=[]))
        self.assertEqual(fallback['policy'],'all_unknown_original_extent')
        self.assertEqual(fallback['bounds'],fallback['asset_bounds'])

    def test_crop_keeps_annotations_and_live_pose_without_changing_local_request(self):
        cells=np.full((100,100),-1,np.int8);cells[30:50,40:60]=0
        grid=GridSnapshot(dict(self.grid.metadata,shape=[100,100]),dict(occupancy=cells))
        value=dict(scope='simulation',map_condition='complete',frame_id='sim_odom',floor_id='unit',access={},
            places=[dict(place_id='p',name='object',aliases=[],reference='base_link',pose=[8.,8.])],
            regions=[dict(region_id='r',name='area',polygon=[[1,1],[2,1],[1,2]])])
        catalog=Catalog(value);reference=ReferenceMap.build(catalog,grid)
        self.assertTrue(reference.record['display_crop']['annotation_expanded'])
        bounds=reference.record['bounds']
        self.assertLessEqual(bounds[0],1.);self.assertGreaterEqual(bounds[2],8.)
        request=[7.,7.,9.,9.]
        view=SpatialView.build(catalog,dict(global_grid=grid,scene=None,pose=[12.,3.,0.],stamp_s=10.,clock_domain='synthetic'),bounds=request)
        self.assertTrue(view.record['display_crop']['live_positions_expanded'])
        self.assertGreater(view.record['overview_bounds'][2],12.)
        self.assertEqual(view.record['full_bounds'],[0.,0.,10.,10.])
        self.assertEqual(view.record['local_bounds'],request)

    def test_narrow_legend_wrap_preserves_chinese_names(self):
        text = 'P1 大厅门口    P2 这是一个用于验证窄图例完整换行的中文地标名称'
        font = map_font(32)
        lines = wrap_map_text(text,font,260)
        self.assertGreater(len(lines),1)
        self.assertEqual(''.join(lines).replace(' ',''),text.replace(' ',''))
        self.assertTrue(all(font.getlength(line)<=260 for line in lines))

    @unittest.skipUnless(Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').is_file(),'Noto CJK font unavailable')
    def test_noto_collection_uses_simplified_chinese_face(self):
        self.assertEqual(map_font(24).getname()[0],'Noto Sans CJK SC')

    def test_every_visible_point_has_a_label_leader(self):
        from unittest.mock import patch
        value=dict(scope='simulation',map_condition='complete',frame_id='sim_odom',floor_id='unit',access={},
            places=[dict(place_id='a',name='门1',reference='base_link',pose=[2.,2.]),
                    dict(place_id='b',name='点2',reference='base_link',pose=[8.,7.])],regions=[])
        catalog=Catalog(value);original=ImageDraw.ImageDraw.line;leaders=[]
        def record(draw,points,*args,**kwargs):
            if kwargs.get('fill')=='#a41b3d':leaders.append(points)
            return original(draw,points,*args,**kwargs)
        with patch.object(ImageDraw.ImageDraw,'line',record):
            render(catalog,self.grid,[0.,0.,14.,10.],None,size=(896,768),detailed=True)
        self.assertEqual(len(leaders),2)
        self.assertTrue(all(len(line)==2 and line[0]!=line[1] for line in leaders))

    def test_dense_point_and_region_labels_are_separate_with_leaders(self):
        entries=[dict(anchor=p,text=t,color='#a41b3d',point=True) for p,t in [
            ((390,280),'P1 电梯1'),((450,280),'P2 电梯2'),((390,330),'P3 电梯3'),((450,330),'P4 电梯4')]]
        entries.append(dict(anchor=(390,270),text='R1 电梯区',color='#14723b',point=False,
            region_bounds=np.array([390,270,450,350])))
        layout=semantic_label_layout(entries,map_font(20),(0,0,800,600),1.)
        self.assertEqual(len(layout),5)
        for i,a in enumerate(layout):
            self.assertNotEqual(a['anchor'],a['end'])
            for b in layout[i+1:]:
                x,y=a['box'],b['box']
                self.assertFalse(x[0]<y[2] and x[2]>y[0] and x[1]<y[3] and x[3]>y[1])
                def cross(p,q,r):return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
                self.assertFalse(cross(a['anchor'],a['end'],b['anchor'])*cross(a['anchor'],a['end'],b['end'])<0 and
                                 cross(b['anchor'],b['end'],a['anchor'])*cross(b['anchor'],b['end'],a['end'])<0)

    def test_region_label_also_has_a_leader(self):
        from unittest.mock import patch
        value=dict(scope='simulation',map_condition='complete',frame_id='sim_odom',floor_id='unit',access={},places=[],
            regions=[dict(region_id='r',name='等候区',polygon=[[2.,2.],[4.,2.],[4.,4.],[2.,4.]])])
        original=ImageDraw.ImageDraw.line;leaders=[]
        def record(draw,points,*args,**kwargs):
            if kwargs.get('fill')=='#14723b':leaders.append(points)
            return original(draw,points,*args,**kwargs)
        with patch.object(ImageDraw.ImageDraw,'line',record):
            render(Catalog(value),self.grid,[0.,0.,14.,10.],None,size=(896,768),detailed=True)
        self.assertEqual(len(leaders),1)
        self.assertNotEqual(leaders[0][0],leaders[0][1])

    def test_many_annotations_can_keep_a_wider_canvas_instead_of_truncating_names(self):
        captions = ['Pale: static prior | blue: measured free / trajectory | grey: unknown',
                    'Orange: robot | red: points / people | green: regions | purple: landmarks',
                    '    '.join(f'P{i} 这是一个完整的地标名称' for i in range(75))]
        layout = reference_layout([0,0,9,15],captions)
        self.assertEqual(layout['size'],(2048,2048))
        self.assertEqual(''.join(layout['lines']).replace(' ',''),''.join(captions).replace(' ',''))

    def test_keyframe_requires_consumed_current_rgb_and_preserves_source_after_recall(self):
        self.session.prompt()
        identity = self.session.observation.record['observation_id']
        rejected = self.session.call('save_keyframe',observation_id=identity,reason='没有模型上下文')
        self.assertEqual(rejected['status'],'refused')
        context = self.session.context()
        self.nav.state['stamp_s'] = 14.
        rejected = self.session.call('save_keyframe',_context=context,observation_id=identity,reason='过期观察')
        self.assertEqual(rejected['status'],'refused')
        saved = self.save()
        self.nav.state['stamp_s'] = 15.
        result = self.session.call('read_keyframe',keyframe_id=saved['keyframe_id'])
        self.assertEqual(result['keyframe'],saved)
        self.assertEqual(saved['source_stamp_s'],14.)
        self.assertTrue(saved['historical_only'])
        self.assertFalse(saved['proves_arrival'])
        self.assertIsNone(self.session.skill.record['handle'])

    def test_saved_images_are_bounded_recallable_and_do_not_count_base64_as_history_text(self):
        saved = []
        for stamp in (10.,11.,12.):
            self.nav.state['stamp_s'] = stamp
            saved.append(self.save())
        body = self.request()
        self.assertEqual(len(self.session.prompt()['keyframes']),2)
        self.assertEqual(sum(p['type']=='image_url' for m in body['messages'] if isinstance(m.get('content'),list)
                             for p in m['content']),6)
        self.assertNotIn('data:image',json.dumps(self.session.conversation.messages()))
        self.assertEqual(len(self.session.keyframes.index(1)),3)
        response = reply('read_keyframe',keyframe_id=saved[0]['keyframe_id'])
        result = self.session.call('read_keyframe',**response.action['arguments'])
        self.session.conversation.commit(response,result,keyframe=result['keyframe'])
        self.assertIn(saved[0]['keyframe_id'],self.session.prompt()['keyframes'])
        self.session.conversation.max_exchanges = 1
        self.session.conversation.notice('trim_fixture')
        self.assertEqual(self.session.prompt()['keyframes'],{})
        self.assertEqual(len(self.session.keyframes.index(1)),3)

    def test_tampered_keyframe_and_other_task_revision_are_rejected(self):
        saved = self.save()
        with self.store.db:
            self.store.db.execute('UPDATE keyframes SET image=? WHERE keyframe_id=?',(b'changed',saved['keyframe_id']))
        with self.assertRaises(ValueError):self.session.keyframes.read(saved['keyframe_id'],1)
        self.session.event('replace','新任务')
        self.assertEqual(self.session.call('read_keyframe',keyframe_id=saved['keyframe_id'])['status'],'unavailable')

    def test_explicit_marker_includes_static_image_and_current_tail_is_unmarked(self):
        body = self.request()
        marked = cache_messages(body['messages'][0]['content'],body['messages'][1:],'explicit')
        self.assertEqual(marked[1]['content'][-1]['cache_control'],dict(type='ephemeral'))
        self.assertNotIn('cache_control',json.dumps(marked[-1]))
        self.assertNotIn('cache_control',json.dumps(body))

    def test_budget_accepts_six_real_images_and_rejects_extra_images(self):
        self.save();self.nav.state['stamp_s']=11.;self.save()
        body = self.request()
        self.assertLess(input_bound(body['messages'],body['tools']),131072)
        oversized = copy.deepcopy(body['messages'])
        oversized[-1]['content'].append(copy.deepcopy(body['messages'][1]['content'][1]))
        with self.assertRaises(ValueError):input_bound(oversized,body['tools'])


class TrajectoryTests(unittest.TestCase):
    def test_matching_frame_name_cannot_replace_the_required_map_identity(self):
        trail = Trajectory()
        state = dict(stamp_s=10.,clock_domain='source',frame_id='map',pose=[1.,2.,0.],
                     localization_status='valid',map_binding=dict(valid=True,map_id='wrong_map'))
        trail.update(state,'map',map_snapshot_id='expected_map')
        self.assertEqual(trail.segments,[])
        state['map_binding']['map_id']='expected_map'
        trail.update(state,'map',map_snapshot_id='expected_map:scan_fusion:asset')
        self.assertEqual(len(trail.segments),1)
        state['map_binding']['valid']=False
        trail.update(state,'map',map_snapshot_id='expected_map:scan_fusion:asset')
        self.assertEqual(trail.segments,[])

    def test_invalid_localization_gaps_and_identity_changes_never_join_segments(self):
        trail = Trajectory(capacity=4)
        state = dict(stamp_s=10.,clock_domain='source',session_id='one',generation=0,frame_id='map',
                     pose=[0.,0.,0.],localization_status='valid',source='measured_fixture')
        trail.update(state,'map')
        state.update(stamp_s=11.,pose=[1.,0.,0.]);trail.update(state,'map')
        state.update(stamp_s=12.,localization_status='invalid');trail.update(state,'map')
        state.update(stamp_s=13.,localization_status='valid',pose=[3.,0.,0.]);trail.update(state,'map')
        state.update(stamp_s=14.,generation=1,pose=[4.,0.,0.]);trail.update(state,'map')
        segments = trail.snapshot(frame_id='map',clock_domain='source',now=13.5)
        self.assertEqual([len(s['samples']) for s in segments],[2,1])
        state.update(stamp_s=20.,pose=[10.,0.,0.]);trail.update(state,'map')
        samples = [p for s in trail.snapshot(frame_id='map',clock_domain='source',now=20.) for p in s['samples']]
        self.assertEqual(len(samples),4)
        self.assertEqual([p['stamp_s'] for p in samples],[11.,13.,14.,20.])
        state.update(stamp_s=21.,map_binding={'map_id':'different'});trail.update(state,'map')
        self.assertEqual(len(trail.segments),1)
