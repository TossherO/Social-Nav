"""Qt integration: asynchronous saves, scope changes and keyboard behavior."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
import tempfile
import time
import unittest

from PyQt5 import QtCore, QtTest, QtWidgets

from decision.vlm.annotation.app import Window
from decision.vlm.annotation.backend import LocalBackend
from decision.vlm.annotation.store import Store, new_event
from tests.test_vlm_annotation import fixture


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.settings = tempfile.TemporaryDirectory()
        QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
        QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, cls.settings.name)

    @classmethod
    def tearDownClass(cls):
        cls.settings.cleanup()

    def wait(self, predicate, seconds=10):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            self.app.processEvents()
            if predicate():
                return
            QtTest.QTest.qWait(10)
        self.fail('Qt操作超时：'+self.window.message.text())

    def idle(self):
        self.wait(lambda: self.window.active_job is None and not self.window.jobs and self.window.fetch_job is None)

    def drag_time(self, start, end):
        from PyQt5 import QtGui
        view=self.window.timeline
        a,b=QtCore.QPointF(view.xpos(start),45),QtCore.QPointF(view.xpos(end),45)
        for kind,point,button,buttons in (
                (QtCore.QEvent.MouseButtonPress,a,QtCore.Qt.LeftButton,QtCore.Qt.LeftButton),
                (QtCore.QEvent.MouseMove,b,QtCore.Qt.NoButton,QtCore.Qt.LeftButton),
                (QtCore.QEvent.MouseButtonRelease,b,QtCore.Qt.LeftButton,QtCore.Qt.NoButton)):
            self.app.sendEvent(view,QtGui.QMouseEvent(kind,point,button,buttons,QtCore.Qt.NoModifier))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)/'project'
        fixture(self.path)
        store = Store(self.path)
        store.writer('seed', acquire=True)
        event = new_event(store.sources()[0], 0, 3, '测试事件')
        event['split'] = 'development'
        store.commit(dict(project_id=store.identity, writer_id='seed', operation_id='seed', expected_revision=0, event=event))
        store.writer('seed', release=True)
        self.window = Window(lambda: LocalBackend(self.path))
        self.assertTrue(Path(self.window.settings.fileName()).is_relative_to(Path(self.settings.name)))
        self.window.show()
        self.wait(lambda: self.window.view_data is not None)
        self.window.goto(20)
        self.wait(lambda: self.window.view_data is not None and self.window.view_data['frame']['id']=='source/20')
        self.idle()

    def tearDown(self):
        self.idle()
        self.window.close()
        self.wait(lambda: self.window.final_close)
        self.app.processEvents()
        self.temporary.cleanup()

    def test_group_review_and_rapid_edits_keep_latest_state(self):
        w = self.window
        w.pick('track:A', False)
        w.pick('track:B', True)
        w.apply_group()
        w.modify_label(lambda label: label.update(note='first'))
        w.modify_label(lambda label: label.update(note='latest'))
        self.idle()
        self.assertFalse(w.paused, w.message.text())
        self.assertEqual(w.label()['note'], 'latest')
        w.mode.setCurrentIndex(1)
        self.wait(lambda: w.view_data is not None)
        w.mark('complete')
        self.idle()
        result = Store(self.path).events()[0]['document']['t1']['source/20']
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['note'], 'latest')
        self.assertEqual(result['groups'][0]['relation'], 'w')

    def test_outside_people_are_normal_targets_and_camera_view_is_context_only(self):
        w = self.window
        binding = w.view_data['window']['fingerprint']
        original = list(w.view_data['histories'])
        fov = dict(origin_xy_m=[.4, .2], boundary_directions_xy=[[.8, .6], [.8, -.6]], forward_xy=[1., 0.])
        inside = dict(key='inside',kind='track',eligible=False,rgb_box=[10.,20.,40.,100.],
                      box=[5.,1.,0.,.7,.7,1.7,0.,.9])
        outside = dict(inside,key='outside',eligible=True,observed=True,confirmed=True,
                       rgb_box=None,box=[-5.,1.,0.,.7,.7,1.7,0.,.9])
        w.view_data['frame'] = dict(w.view_data['frame'], camera_fov=fov,
                                    people=w.view_data['frame']['people']+[inside,outside])
        for row in w.view_data['context_frames']:
            row['frame'] = dict(row['frame'],camera_fov=fov,people=w.view_data['frame']['people'])
        w.view_data['window'] = dict(w.view_data['window'],mapping=dict(w.view_data['window']['mapping'],outside=2))
        w.render_scene()
        self.assertEqual({p['key'] for p in w.scene_people()}, {'track:A','track:B','inside','outside'})
        self.assertEqual(w.people.count(), 4)
        self.assertNotIn('参考',w.people.item(3).text())
        self.assertIn('无RGB投影',w.people.item(3).text())
        for card in w.history:
            self.assertIn('outside',card.context_bev.items_by_key)
            self.assertNotIn('inside',card.context_bev.items_by_key)
            self.assertEqual(card.context_bev.items_by_key['outside'].pen().style(),QtCore.Qt.SolidLine)
            self.assertEqual(card.context_bev.texts['outside'].toPlainText(),'P2')
            self.assertIn('相机水平视野',card.context_bev.bev_title.toPlainText())
            fills = [item for item in card.context_bev.scene().items()
                     if isinstance(item,QtWidgets.QGraphicsPolygonItem) and item.parentItem() is not None]
            self.assertEqual(len(fills), 1)
            self.assertEqual(fills[0].brush().color().name(), '#79aaff')
            self.assertLess(fills[0].brush().color().alpha(), 30)
            self.assertTrue(fills[0].parentItem().flags() & QtWidgets.QGraphicsItem.ItemClipsChildrenToShape)
        self.assertEqual(w.view_data['histories'],original)
        self.assertEqual(w.view_data['window']['fingerprint'],binding)
        w.pick('outside',False)
        self.assertEqual(w.eligible_selection(),['outside'])
        self.assertNotIn('outside',w.rgb.items_by_key)
        w.history[2].context_bev.show_frame(None,None,[],{})
        self.assertIn('相机视野不可用',w.history[2].context_bev.bev_title.toPlainText())

    def test_delete_event_confirms_scope_clears_ui_and_allows_new_full_event(self):
        from unittest.mock import patch
        w=self.window
        w.modify_label(lambda label:label.update(note='将被删除的标注'))
        self.idle()
        ident=w.event['id']
        original=(self.path/'observations.sqlite').read_bytes()
        w.event_settings()
        self.assertTrue(w.event_delete.isVisible())
        with patch.object(QtWidgets.QMessageBox,'question',return_value=QtWidgets.QMessageBox.No) as confirm:
            w.event_delete.click()
        self.assertIn('1 个T1样本',confirm.call_args.args[2])
        self.assertEqual(confirm.call_args.args[-1],QtWidgets.QMessageBox.No)
        self.assertEqual(len(Store(self.path).events()),1)
        self.assertTrue(w.event_editor.isVisible())
        with patch.object(QtWidgets.QMessageBox,'question',return_value=QtWidgets.QMessageBox.Yes):
            w.event_delete.click()
            self.assertEqual(w.deleting,ident)
            self.assertFalse(w.event_delete.isEnabled())
            self.assertFalse(w.event_apply.isEnabled())
        self.idle()
        self.assertEqual(Store(self.path).events(),[])
        self.assertEqual(w.snapshot_data['events'],[])
        self.assertIsNone(w.event)
        self.assertIsNone(w.events.currentData())
        self.assertEqual(w.events.count(),1)
        self.assertFalse(w.event_editor.isVisible())
        self.assertFalse(w.editable())
        self.assertEqual((self.path/'observations.sqlite').read_bytes(),original)
        w.undo();self.idle()
        self.assertEqual(Store(self.path).events(),[])
        w.create_full_event();self.idle()
        self.assertNotEqual(w.event['id'],ident)
        self.assertEqual(w.event['t1'],{})
        w.writable=False;w.event_settings()
        self.assertFalse(w.event_delete.isEnabled())
        with patch.object(QtWidgets.QMessageBox,'question') as confirm:
            w.delete_event()
        confirm.assert_not_called()
        w.cancel_event_editor()

    def test_delete_waits_for_queued_save_and_preserves_new_selection(self):
        import threading
        from unittest.mock import patch
        w=self.window
        other=new_event(w.source(),0,3,'保留事件')
        other['split']='development'
        result=w.backend.commit(other,0)
        w.revisions[other['id']]=result['revision']
        w.edit_serial[other['id']]=0
        w.update_snapshot(other)
        w.events.addItem(other['name'],other['id'])
        ident=w.event['id']
        released=threading.Event()
        delete=w.backend.delete_event
        observed=[]
        def delayed(event_id,revision):
            observed.append((event_id,revision))
            if not released.wait(5):
                raise AssertionError('delete test was not released')
            return delete(event_id,revision)
        try:
            with patch.object(w.backend,'delete_event',side_effect=delayed), patch.object(
                    QtWidgets.QMessageBox,'question',return_value=QtWidgets.QMessageBox.Yes):
                w.modify_label(lambda label:label.update(note='先保存再删除'))
                w.event_settings();w.delete_event()
                w.select_event(other['id'])
                self.assertFalse(w.editable())
                self.wait(lambda:bool(observed))
                self.assertEqual(observed,[(ident,2)])
                released.set()
                self.idle()
        finally:
            released.set()
        self.assertEqual(w.event['id'],other['id'])
        self.assertTrue(w.editable())
        self.assertEqual([row['document']['id'] for row in Store(self.path).events()],[other['id']])
        self.assertEqual(w.events.findData(ident),-1)

    def test_delete_conflict_keeps_event_and_pauses_writes(self):
        from unittest.mock import patch
        from decision.vlm.annotation.store import Conflict
        w=self.window
        ident=w.event['id']
        w.event_settings()
        with patch.object(w.backend,'delete_event',side_effect=Conflict('事件已被修改')), patch.object(
                QtWidgets.QMessageBox,'question',return_value=QtWidgets.QMessageBox.Yes):
            w.delete_event()
            self.idle()
        self.assertIsNone(w.deleting)
        self.assertTrue(w.paused)
        self.assertEqual(w.event['id'],ident)
        self.assertEqual(Store(self.path).events()[0]['document']['id'],ident)
        self.assertTrue(w.event_editor.isVisible())
        self.assertFalse(w.event_delete.isEnabled())
        self.assertIn('已被修改',w.message.text())
        # Recover from the explicit simulated conflict before closing this fixture.
        with patch.object(QtWidgets.QMessageBox,'question',return_value=QtWidgets.QMessageBox.Yes):
            w.reconnect()
        self.idle()

    def test_timeline_drag_creates_inline_event_only_after_enter_and_cancel_keeps_it(self):
        w=self.window
        old=w.event['id']
        w.t1_note.setText('保留已有备注')
        w.t1_note.setFocus()
        self.drag_time(.5,1.5)
        self.wait(lambda:w.view_data is not None and w.frame_index==15)
        self.idle()
        self.assertTrue(w.event_editor.isVisible())
        self.assertIsNone(self.app.activeModalWidget())
        self.assertEqual((w.event_start.value(),w.event_end.value()),(.5,1.5))
        records=Store(self.path).events()
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]['document']['t1']['source/20']['note'],'保留已有备注')
        w.event_name.setText('同行片段')
        w.event_name.setFocus()
        QtTest.QTest.keyClick(w.event_name,QtCore.Qt.Key_Return)
        self.idle()
        self.assertFalse(w.event_editor.isVisible())
        self.assertEqual(len(Store(self.path).events()),2)
        self.assertNotEqual(w.event['id'],old)
        self.assertEqual((w.event['start'],w.event['end'],w.event['name']),(.5,1.5,'同行片段'))
        self.assertEqual(w.event['t1'],{})
        w.event_settings()
        w.event_end.setValue(2)
        QtTest.QTest.keyClick(w.event_end,QtCore.Qt.Key_Escape)
        self.assertFalse(w.event_editor.isVisible())
        self.assertEqual(w.event['end'],1.5)
        self.assertIsNone(w.timeline.selection)
        w.event_settings();w.event_end.setValue(2)
        w.event_cancel.setFocus()
        QtTest.QTest.keyClick(w.event_cancel,QtCore.Qt.Key_Return)
        self.assertFalse(w.event_editor.isVisible())
        self.assertEqual(w.event['end'],1.5)
        created=w.event['id']
        w.select_event(old)
        point=QtCore.QPoint(round(w.timeline.xpos(1)),10)
        self.assertEqual(w.timeline.event_at(point.x())['id'],created)
        QtTest.QTest.mouseClick(w.timeline,QtCore.Qt.LeftButton,pos=point)
        self.wait(lambda:w.view_data is not None and w.frame_index==10)
        self.assertEqual(w.event['id'],created)
        QtTest.QTest.mouseDClick(w.timeline,QtCore.Qt.LeftButton,pos=point)
        self.assertEqual(w.event_editor_id,created)
        self.assertTrue(w.event_editor.isVisible())
        w.cancel_event_editor()

    def test_i_o_range_shortcuts_and_full_bag_creation_reuse_existing_event(self):
        w=self.window
        old=w.event['id']
        w.limit_event.setChecked(True)
        w.goto(5);self.wait(lambda:w.view_data is not None and w.frame_index==5)
        w.people.setFocus();QtTest.QTest.keyClick(w.people,QtCore.Qt.Key_I)
        self.assertTrue(w.event_editor.isVisible())
        self.assertEqual(w.event_start.value(),.5)
        w.interval.setValue(2)
        w.event_start.setFocus();QtTest.QTest.keyClick(w.event_start,QtCore.Qt.Key_S)
        self.wait(lambda:w.view_data is not None and w.frame_index==25)
        QtTest.QTest.keyClick(w.event_start,QtCore.Qt.Key_O)
        self.assertEqual((w.event_start.value(),w.event_end.value()),(.5,2.5))
        w.event_name.setFocus();QtTest.QTest.keyClicks(w.event_name,'indoor')
        self.assertEqual((w.event_start.value(),w.event_end.value()),(.5,2.5))
        QtTest.QTest.keyClick(w.event_name,QtCore.Qt.Key_Return)
        self.idle()
        self.assertEqual(w.event['name'],'indoor')
        w.create_full_event();self.idle()
        self.assertEqual(w.event['id'],old)
        self.assertEqual(len(Store(self.path).events()),2)
        w.create_full_event();self.idle()
        self.assertEqual(len(Store(self.path).events()),2)
        w.resize(1280,800);w.event_settings();self.app.processEvents()
        w.position.setText('68.900s · 非标注样本 · 检测输入3/3帧 · 临时选区，仅浏览')
        self.app.processEvents()
        self.assertLessEqual(w.height(),850)
        self.assertLessEqual(w.width(),1280)
        w.cancel_event_editor()

    def test_inline_range_edits_protect_labels_and_global_browsing_cannot_write_outside(self):
        w=self.window
        w.modify_label(lambda label:label.update(note='已有标注'));self.idle()
        w.event_settings();w.event_end.setValue(1)
        w.commit_event_editor()
        self.assertTrue(w.event_editor.isVisible())
        self.assertIn('排除',w.message.text())
        self.assertEqual(w.event['end'],3)
        w.cancel_event_editor()
        w.event_settings();self.drag_time(0,1)
        self.assertEqual((w.event_start.value(),w.event_end.value()),(1,3))
        w.event_end.setValue(2.2)
        w.event_name.setText('范围调整')
        w.commit_event_editor();self.idle()
        self.assertEqual((w.event['start'],w.event['end']),(1,2.2))
        self.assertEqual(w.events.currentText(),'范围调整')
        self.assertEqual(w.event['t1']['source/20']['note'],'已有标注')
        w.jump_time(.5);self.wait(lambda:w.view_data is not None and w.frame_index==5)
        self.assertFalse(w.editable())
        self.assertIn('事件外',w.position.text())
        w.mark('skipped');self.idle()
        self.assertEqual(set(Store(self.path).events()[0]['document']['t1']),{'source/20'})
        w.limit_event.setChecked(True)
        self.wait(lambda:w.view_data is not None and w.frame_index==10)
        w.navigate(-1)
        self.assertEqual(w.frame_index,10)
        w.undo();self.idle()
        self.assertEqual((w.event['start'],w.event['end']),(0,3))
        self.assertEqual(w.event['t1']['source/20']['note'],'已有标注')
        w.redo();self.idle()
        self.assertEqual((w.event['start'],w.event['end']),(1,2.2))
        original=w.event['id']
        w.create_full_event();self.idle()
        self.assertEqual((w.event['start'],w.event['end']),(0,3))
        self.assertEqual(w.event['split'],'development')
        self.assertEqual(w.event['t1'],{})
        records=Store(self.path).events()
        self.assertEqual(len(records),2)
        self.assertEqual(next(r['document'] for r in records if r['document']['id']==original)['t1']['source/20']['note'],'已有标注')
        w.create_full_event();self.idle()
        self.assertEqual(len(Store(self.path).events()),2)

    def test_browse_does_not_create_samples_and_text_keeps_wasd(self):
        w = self.window
        w.people.setFocus()
        QtTest.QTest.keyClick(w.people, QtCore.Qt.Key_D)
        self.wait(lambda: w.view_data is not None and w.view_data['frame']['id']=='source/21')
        self.assertEqual(w.event['t1'], {})
        w.reviewer.setFocus()
        w.reviewer.setText('')
        QtTest.QTest.keyClicks(w.reviewer, 'wasd')
        self.assertEqual(w.reviewer.text(), 'wasd')
        self.assertEqual(w.frame_index, 21)
        w.people.setFocus()
        QtTest.QTest.keyClick(w.people, QtCore.Qt.Key_2)
        self.idle()
        self.assertEqual(w.event['t1']['source/21']['status'], 'skipped')
        self.assertEqual(w.frame_index, 21)

    def test_t2_evidence_page_saves_independently_of_t1(self):
        w = self.window
        def seed(event):
            event['persons']['target-person'] = dict(name='目标A')
            event['links'].append(dict(key='track:A', person_id='target-person', start_ns=1700000000000000000,
                                       end_ns=1700000003000000000))
            event['targets']['task'] = dict(id='task', description='灰色衣服的人', person_id='target-person', absent=False,
                                            reference_frame='source/10', samples={})
        w.mutate(seed)
        self.idle()
        w.refresh_targets()
        w.tabs.setCurrentIndex(1)
        w.pick('track:A', False)
        w.set_true_key()
        w.set_candidates()
        for field, value in (('visibility', 'clear'), ('coverage', 'represented'), ('match', 'unique')):
            widget = w.t2_controls[field]
            widget.setCurrentIndex(widget.findData(value))
        self.idle()
        self.assertFalse(w.paused, w.message.text())
        w.mode.setCurrentIndex(1)
        self.wait(lambda:w.view_data is not None)
        w.mark('complete')
        self.idle()
        event = Store(self.path).events()[0]['document']
        self.assertEqual(event['t1'], {})
        self.assertEqual(event['targets']['task']['samples']['source/20']['status'], 'complete')

    def test_interval_sampling_does_not_build_every_model_window(self):
        from unittest.mock import patch
        w = self.window
        with patch.object(w.backend,'window',side_effect=AssertionError('bulk sampling should stay lightweight')):
            w.batch_samples()
            self.idle()
        self.assertFalse(w.paused,w.message.text())
        self.assertEqual(set(w.event['t1']),{'source/0','source/20'})
        self.assertTrue(all(label['binding'] is None for label in w.event['t1'].values()))

    def test_3d_selection_camera_and_exact_three_frame_review(self):
        w = self.window
        expected = w.backend.review_images('source/20')
        self.assertTrue(all(thumb.data.startswith(b'\xff\xd8') for thumb in w.history))
        self.assertTrue(all(raw.startswith(b'\x89PNG') for raw in expected))
        self.assertEqual(w.layout_choice.currentIndex(),0)
        w.layout_choice.setCurrentIndex(2)
        self.wait(lambda:w.view_data is not None)
        self.idle()
        self.assertEqual(w.context_views.currentIndex(),0)
        w.cloud.show_frame(w.view_data['cloud3d'],w.scene_people()[:1],{'track:A':'P0'})
        w.cloud.repaint()
        self.app.processEvents()
        self.assertTrue(w.cloud.hits)
        point = w.cloud.hits[0][1].center().toPoint()
        QtTest.QTest.mouseClick(w.cloud,QtCore.Qt.LeftButton,pos=point)
        self.assertEqual(w.selected,['track:A'])
        from PyQt5 import QtGui
        point = QtCore.QPointF(150,150)
        for kind,pos,button,buttons in (
                (QtCore.QEvent.MouseButtonPress,point,QtCore.Qt.LeftButton,QtCore.Qt.LeftButton),
                (QtCore.QEvent.MouseMove,point+QtCore.QPointF(60,25),QtCore.Qt.NoButton,QtCore.Qt.LeftButton),
                (QtCore.QEvent.MouseButtonRelease,point+QtCore.QPointF(60,25),QtCore.Qt.LeftButton,QtCore.Qt.NoButton)):
            self.app.sendEvent(w.cloud,QtGui.QMouseEvent(kind,pos,button,buttons,QtCore.Qt.NoModifier))
        azimuth = w.cloud.azimuth
        self.assertNotEqual(azimuth,-140.)
        w.goto(21)
        self.wait(lambda:w.view_data is not None and w.view_data['frame']['id']=='source/21')
        self.idle()
        self.assertEqual(w.cloud.azimuth,azimuth)
        w.point_size.setValue(4)
        self.assertEqual(w.cloud.point_size,4)
        w.start_auxiliary()
        self.assertEqual(w.context_views.currentIndex(),1)
        w.mode.setCurrentIndex(1)
        self.wait(lambda:w.view_data is not None)
        self.idle()
        self.assertEqual(w.visual_stack.currentIndex(),1)
        self.assertIsNone(w.view_data['cloud3d'])
        self.assertEqual(w.audit_view.data,w.history[2].data)
        w.resize(1280,800)
        self.app.processEvents()
        self.assertLessEqual(w.height(),850)
        w.layout_choice.setCurrentIndex(1)
        w.resize(1280,800)
        self.app.processEvents()
        self.assertLessEqual(w.height(),850)

    def test_bev_ids_stay_separate_after_zoom_and_can_select_people(self):
        w = self.window
        w.context_views.setCurrentIndex(2)
        w.layout_choice.setCurrentIndex(2)
        self.wait(lambda:w.view_data is not None)
        self.idle()
        people = w.scene_people()
        people[1]['box'][:2] = people[0]['box'][:2]
        w.bev.show_frame(None,None,people,{'track:A':'P0','track:B':'P1'})
        w.bev.fit()
        for scale in (1.,1.5):
            w.bev.scale(scale,scale)
            w.bev.layout_bev_labels()
            self.app.processEvents()
            rectangles = [text.deviceTransform(w.bev.viewportTransform()).mapRect(text.boundingRect())
                          for text in w.bev.texts.values()]
            self.assertFalse(rectangles[0].intersects(rectangles[1]))
            self.assertTrue(all(w.bev.viewport().rect().contains(rect.toAlignedRect()) for rect in rectangles))
        label = w.bev.texts['track:B']
        point = label.deviceTransform(w.bev.viewportTransform()).map(label.boundingRect().center()).toPoint()
        QtTest.QTest.mouseClick(w.bev.viewport(),QtCore.Qt.LeftButton,pos=point)
        self.assertEqual(w.selected,['track:B'])
        self.assertEqual(w.event['t1'],{})

    def test_current_three_frame_images_pick_people_with_letterboxing_and_resize(self):
        w=self.window
        card=w.history[2]
        original=card.data
        fingerprint=w.view_data['window']['fingerprint']
        for size in ((1600,1000),(1280,800)):
            w.resize(*size)
            self.app.processEvents()
            for view,kind,key,multiple in ((card.rgb,'rgb','track:A',False),(card.context_bev,'bev','track:B',True)):
                if kind=='rgb':
                    region=next(r for r in view.regions if r['kind']==kind and r['key']==key)
                    point=view.image_to_widget(region['rect'].center()).toPoint()
                    target=view
                else:
                    point=view.mapFromScene(view.items_by_key[key].rect().center())
                    target=view.viewport()
                QtTest.QTest.mouseClick(target,QtCore.Qt.LeftButton,
                                      QtCore.Qt.ControlModifier if multiple else QtCore.Qt.NoModifier,point)
                self.assertEqual(w.selected,['track:A','track:B'] if multiple else ['track:A'])
                self.assertEqual([w.people.item(i).data(QtCore.Qt.UserRole) for i in range(w.people.count())
                                  if w.people.item(i).isSelected()],w.selected)
        self.assertEqual(card.data,original)
        self.assertEqual(w.view_data['window']['fingerprint'],fingerprint)
        self.assertEqual(w.event['t1'],{})
        QtTest.QTest.mouseClick(card.rgb,QtCore.Qt.LeftButton,pos=QtCore.QPoint(1,1))
        self.assertEqual(w.selected,['track:A','track:B'])
        self.assertFalse(w.history[0].rgb.regions)
        self.assertFalse(w.history[1].bev.regions)
        QtTest.QTest.mouseClick(w.history[0].rgb,QtCore.Qt.LeftButton,pos=QtCore.QPoint(100,100))
        self.assertEqual(w.selected,['track:A','track:B'])
        # Exercise the uncached path; buffered navigation now finishes synchronously.
        cached=w.view_cache.pop(('source/21',False),None)
        if cached:
            w.view_cache_bytes-=cached[1]
        w.goto(21)
        self.assertFalse(card.rgb.regions)
        self.wait(lambda:w.view_data is not None and w.frame_index==21)
        w.reviewer.setText('测试者')
        w.reviewer.setFocus()
        region=next(r for r in card.rgb.regions if r['kind']=='rgb' and r['key']=='track:A')
        QtTest.QTest.mouseClick(card.rgb,QtCore.Qt.LeftButton,pos=card.rgb.image_to_widget(region['rect'].center()).toPoint())
        QtTest.QTest.keyClick(w.windowHandle(),QtCore.Qt.Key_D)
        self.assertEqual(w.frame_index,22)
        self.assertEqual(w.reviewer.text(),'测试者')
        self.wait(lambda:w.view_data is not None and w.frame_index==22)

    def test_group_display_survives_save_redraw_undo_and_frame_changes(self):
        w=self.window
        original=w.history[2].data
        w.pick('track:A',False)
        w.pick('track:B',True)
        w.apply_group()
        self.idle()
        self.assertEqual(w.groups.count(),1)
        self.assertIn('G1',w.group_summary.text())
        self.assertIn('P0',w.group_summary.text())
        self.assertIn('P1',w.group_summary.text())
        self.assertTrue(all('G1' in w.people.item(i).text() for i in range(2)))
        colors=w.history[2].context_bev.group_colors
        self.assertEqual(colors['track:A'],colors['track:B'])
        w.render_scene()
        self.assertTrue(all('G1' in w.people.item(i).text() for i in range(2)))
        self.assertEqual(w.history[2].data,original)
        w.undo();self.idle()
        self.assertNotIn('G1',w.group_summary.text())
        self.assertFalse(w.history[2].context_bev.group_colors)
        w.redo();self.idle()
        self.assertIn('G1',w.group_summary.text())
        w.goto(21)
        self.wait(lambda:w.view_data is not None and w.frame_index==21)
        self.assertNotIn('G1',w.group_summary.text())
        self.assertFalse(w.history[2].context_bev.group_colors)
        w.goto(20)
        self.wait(lambda:w.view_data is not None and w.frame_index==20)
        self.assertIn('G1',w.group_summary.text())

    def test_review_images_remain_neutral_while_current_people_are_selectable(self):
        w=self.window
        w.pick('track:A',False);w.pick('track:B',True)
        w.apply_group();self.idle()
        original=w.backend.review_images('source/20')[2]
        w.mode.setCurrentIndex(1)
        self.wait(lambda:w.view_data is not None)
        self.idle()
        thumb=w.history[2].bev
        self.assertFalse(thumb.show_overlays)
        self.assertIn('G1',w.group_summary.text())
        before=thumb.grab().toImage()
        region=next(r for r in thumb.regions if r['kind']=='bev' and r['key']=='track:A')
        QtTest.QTest.mouseClick(thumb,QtCore.Qt.LeftButton,pos=thumb.image_to_widget(region['rect'].center()).toPoint())
        self.app.processEvents()
        self.assertEqual(w.selected,['track:A'])
        self.assertEqual(thumb.grab().toImage(),before)
        self.assertEqual(w.history[2].data,original)
        self.assertFalse(w.audit_view.show_overlays)

    def test_cached_navigation_and_playback_do_not_wait_for_background_io(self):
        import threading
        from unittest.mock import patch
        w=self.window
        self.assertIn(('source/21',False),w.view_cache)
        with patch.object(w,'schedule_prefetch'),patch.object(w.backend,'views',side_effect=AssertionError('cached switch must not read')):
            w.goto(21)
            self.assertEqual(w.view_data['frame']['id'],'source/21')
            w.goto(20)
            self.assertEqual(w.view_data['frame']['id'],'source/20')
        released=threading.Event()
        try:
            w.task('模拟后台I/O',lambda:released.wait(5))
            w.toggle_play()
            with patch('decision.vlm.annotation.app.time.monotonic',return_value=100):
                w.play_tick()
            with patch('decision.vlm.annotation.app.time.monotonic',return_value=100.11):
                w.play_tick()
            self.assertEqual(w.frame_index,21)
            self.assertIsNotNone(w.active_job)
            self.assertTrue(w.playback.isActive())
        finally:
            w.playback.stop()
            released.set()
        self.idle()
        self.assertEqual(Store(self.path).events()[0]['document']['t1'],{})

    def test_prefetch_includes_backward_and_actual_sample_targets(self):
        from unittest.mock import patch
        w=self.window
        targets=[row['id'] for row in w.prefetch_rows()]
        self.assertEqual(targets[:4],['source/21','source/19','source/30','source/0'])
        w.interval.setValue(1);self.idle()
        self.assertIn(('source/10',False,w.point_size.value()),w.prepared_cache)
        with patch.object(w,'schedule_prefetch'),patch.object(w.backend,'views',side_effect=AssertionError('sample target should be ready')):
            w.navigate_sample(-1)
            self.assertEqual(w.view_data['frame']['id'],'source/10')
        self.assertEqual(w.event['t1'],{})

    def test_recently_visited_frame_survives_idle_prefetch_pressure(self):
        from unittest.mock import patch
        from collections import OrderedDict
        w=self.window
        w.goto(19);self.wait(lambda:w.view_data is not None);self.idle()
        w.goto(20);self.wait(lambda:w.view_data is not None);self.idle()
        saved=w.prepared_cache;size=w.point_size.value()
        w.prepared_cache=OrderedDict((('source/'+str(i),False,size),saved[('source/'+str(i),False,size)]) for i in (19,21,22,23,24,25,20))
        w.prepared_bytes=sum(row[1] for row in w.prepared_cache.values());budget=w.prepared_bytes
        try:
            with patch('decision.vlm.annotation.app.PREPARED_CACHE_BYTES',budget):
                w.cache_prepared(w.view_cache[('source/26',False)][0],size,saved[('source/26',False,size)][0])
            self.assertIn(('source/19',False,size),w.prepared_cache)
            self.assertNotIn(('source/21',False,size),w.prepared_cache)
            self.assertLessEqual(w.prepared_bytes,budget)
        finally:
            w.prepared_cache=saved;w.prepared_bytes=sum(row[1] for row in saved.values())

    def test_reconnect_keeps_same_input_cache_and_cursor_but_clears_changed_identity(self):
        import copy
        from unittest.mock import patch
        w=self.window
        cached=w.view_cache[('source/20',False)][0]
        snapshot=w.backend.snapshot()
        with patch.object(w,'schedule_prefetch'),patch.object(w.backend,'views',side_effect=AssertionError('same project cache should survive reconnect')):
            w.connected(snapshot);self.idle()
        self.assertEqual(w.frame_index,20)
        self.assertIs(w.view_cache[('source/20',False)][0],cached)
        changed=copy.deepcopy(snapshot);changed['display_implementation']='different-display-source'
        with patch.object(w,'source_changed'):
            w.connected(changed)
        self.assertFalse(w.view_cache)
        self.assertFalse(w.prepared_cache)
        self.assertFalse(w.recent_views)

    def test_failed_frame_read_replaces_loading_state(self):
        from unittest.mock import patch
        w=self.window
        raw=w.view_cache.pop(('source/10',False),None)
        if raw:w.view_cache_bytes-=raw[1]
        prepared=w.prepared_cache.pop(('source/10',False,w.point_size.value()),None)
        if prepared:w.prepared_bytes-=prepared[1]
        with patch.object(w.backend,'views',side_effect=ValueError('模拟连接失败')):
            w.goto(10);self.idle()
        self.assertIsNone(w.view_data)
        self.assertIn('读取失败',w.position.text())
        self.assertIn('模拟连接失败',w.message.text())

    def test_playback_buffers_without_skipping_or_mislabeling_old_frame(self):
        from unittest.mock import patch
        w=self.window
        old=w.view_data
        w.view_cache.clear();w.view_cache_bytes=0
        with patch.object(w,'schedule_prefetch'):
            w.toggle_play();w.play_tick()
            self.assertTrue(w.play_buffering)
            self.assertEqual(w.frame_index,20)
            self.assertIs(w.view_data,old)
            self.assertIn('缓冲',w.message.text())
            self.assertIsNone(w.play_anchor)
            w.playback.stop()
        with patch.object(w.history[2].rgb,'show_original') as show:
            w.inspect_observation(2)
            self.idle()
            self.assertEqual(show.call_args.args[0],w.backend.review_images('source/20')[2])

    def test_browse_preview_does_not_rebind_or_falsely_invalidate_reviewed_labels(self):
        w=self.window
        w.pick('track:A',False);w.pick('track:B',True)
        w.apply_group();self.idle()
        self.assertIsNone(w.label()['binding'])
        w.mode.setCurrentIndex(1)
        self.wait(lambda:w.view_data is not None);self.idle()
        w.mark('complete');self.idle()
        binding=w.label()['binding']
        self.assertTrue(binding)
        w.mode.setCurrentIndex(0)
        self.wait(lambda:w.view_data is not None);self.idle()
        self.assertFalse(w.view_data['window']['input_verified'])
        self.assertIsNone(w.view_data['window']['fingerprint'])
        self.assertEqual((w.label()['status'],w.label()['binding']),('complete',binding))
        self.assertNotIn('输入变化',w.position.text())
        w.modify_label(lambda label:label.update(note='实际修改后才取消审核'))
        self.idle()
        self.assertEqual(w.label()['status'],'draft')
        self.assertIsNone(w.label()['binding'])

    def test_playback_eviction_keeps_next_frame_ahead_of_recently_displayed_past(self):
        from unittest.mock import patch
        from collections import OrderedDict
        w=self.window
        size=w.point_size.value()
        w.goto(19);self.wait(lambda:w.view_data is not None);self.idle()
        w.goto(20);self.wait(lambda:w.view_data is not None);self.idle()
        saved=w.prepared_cache
        w.prepared_cache=OrderedDict((('source/'+str(i),False,size),saved[('source/'+str(i),False,size)]) for i in (21,22,23,19))
        w.prepared_bytes=sum(row[1] for row in w.prepared_cache.values())
        budget=w.prepared_bytes
        w.playback.start()
        try:
            with patch('decision.vlm.annotation.app.PREPARED_CACHE_BYTES',budget):
                w.cache_prepared(w.view_cache[('source/24',False)][0],size,saved[('source/24',False,size)][0])
            self.assertIn(('source/21',False,size),w.prepared_cache)
            self.assertNotIn(('source/19',False,size),w.prepared_cache)
            self.assertLessEqual(w.prepared_bytes,budget)
        finally:
            w.playback.stop()
            w.prepared_cache=saved
            w.prepared_bytes=sum(row[1] for row in saved.values())

    def test_bev_height_colors_keep_highest_overlap_independent_of_point_order(self):
        import io
        import numpy as np
        from decision.vlm.annotation.views import PersonView
        view=PersonView(bev=True)
        view.factor=20
        view.setSceneRect(-60,-120,120,160)
        points=np.array([[2.,0.,-1.,1.],[2.,0.,1.,2.],[2.05,0.,.5,3.],[4.,0.,-1.,1.]],dtype='<f4')
        def raster(values):
            stream=io.BytesIO()
            np.save(stream,values,allow_pickle=False)
            return view.cloud_pixmap(stream.getvalue()).toImage()
        image=raster(points)
        self.assertEqual(image,raster(points[::-1]))
        high=raster(points[1:2])
        self.assertEqual(image.pixelColor(60,80),high.pixelColor(60,80))
        self.assertNotEqual(image.pixelColor(60,80),image.pixelColor(60,40))

    def test_fast_context_labels_preserve_collision_layout(self):
        from decision.vlm.annotation.views import place_context_labels
        from decision.vlm.scene_inputs import place_bev_labels
        markers=[(str(i),110+(i%4)*7,90+(i//4)*9) for i in range(17)]
        sizes={key:(24+3*i,22) for i,(key,_,_) in enumerate(markers)}
        bounds=(0,30,250,260)
        self.assertEqual(place_context_labels(markers,sizes,bounds),place_bev_labels(markers,sizes,bounds))

    def test_bev_current_targets_and_confirmed_historical_references_have_distinct_styles(self):
        import copy
        import math
        w=self.window
        view=w.history[2].context_bev
        original=w.history[2].data
        binding=w.view_data['window']['fingerprint']
        reference=dict(key='reference/g0',eligible=False,kind='track',observed=False,confirmed=True,
                       box=[200.,-1.,0.,2.,1.,1.7,math.pi/2,.9],rgb_box=[480.,100.,600.,400.])
        tentative=dict(reference,key='tentative/g0',observed=True,confirmed=False)
        w.view_data['frame']['people'].extend([reference,tentative])
        w.view_data['frame']['people'][0]['box']=[2.,-1.,0.,2.,1.,1.7,math.pi/2,.9]
        current=w.view_data['context_frames'][-1]
        current['frame']=dict(current['frame'],people=w.view_data['frame']['people'])
        current['trajectory']=dict(current['trajectory'],people={'reference/g0':[
            dict(observed=False,points=[[-1.,200.,0.],[0.,201.,0.]])]})
        for row in w.view_data['context_frames'][:-1]:
            historical=dict(w.view_data['frame']['people'][0],eligible=False,observed=False,confirmed=False)
            old_only=dict(reference,key='historical-only',eligible=True,observed=True,
                          box=[5.,-3.,0.,.7,.7,1.7,0.,.9])
            old_predicted=dict(old_only,key='historical-predicted',eligible=False,observed=False,
                               box=[6.,-3.,0.,.7,.7,1.7,0.,.9])
            row['frame']=dict(row['frame'],people=[historical,old_only,old_predicted,tentative])
        # The reference number belongs to the identity, not its order in a frame.
        w.view_data['context_frames'][1]['frame']['people'].reverse()
        raw=copy.deepcopy(w.view_data['context_frames'])
        w.render_scene()
        self.app.processEvents()
        self.assertIsNotNone(view.cloud)
        self.assertEqual(w.history[2].bev_stack.currentWidget(),view)
        self.assertEqual(set(view.items_by_key),{'track:A','track:B'})
        self.assertEqual(view.factor,20.)
        self.assertNotIn('reference/g0',w.history[2].context['trajectory']['people'])
        item=view.items_by_key['track:A']
        self.assertEqual(len(item.polygon),4)
        self.assertAlmostEqual(item.rect().width(),2*view.factor)
        self.assertAlmostEqual(item.rect().height(),view.factor)
        for box in view.items_by_key.values():
            self.assertEqual(box.pen().style(),QtCore.Qt.SolidLine)
            self.assertEqual(box.pen().color().name(),'#55dbea')
        for card in w.history[:2]:
            self.assertEqual(set(card.context_bev.items_by_key),{'track:A','historical-only','historical-predicted'})
            normal=card.context_bev.items_by_key['track:A']
            self.assertEqual(normal.pen().style(),QtCore.Qt.SolidLine)
            self.assertEqual(normal.pen().color().name(),'#55dbea')
            self.assertEqual(card.context_bev.texts['track:A'].toPlainText(),'P0')
            for number,key in enumerate(('historical-only','historical-predicted'),1):
                reference_item=card.context_bev.items_by_key[key]
                self.assertEqual(reference_item.pen().style(),QtCore.Qt.DashLine)
                self.assertEqual(reference_item.pen().color().name(),'#ffad66')
                self.assertEqual(card.context_bev.texts[key].toPlainText(),f'R{number}')
                self.assertNotIn(key,w.view_data['window']['mapping'])
            normal=next(p for p in card.context['frame']['people'] if p['key']=='track:A')
            self.assertFalse(normal['eligible'])
            self.assertFalse(normal['observed'])
            self.assertFalse(normal['confirmed'])
        self.assertEqual(w.view_data['context_frames'],raw)
        self.assertEqual(w.view_data['window']['fingerprint'],binding)
        self.assertEqual({p['key'] for p in w.scene_people()}, {'track:A','track:B','reference/g0','tentative/g0'})
        QtTest.QTest.mouseClick(view.viewport(),QtCore.Qt.LeftButton,pos=view.mapFromScene(item.rect().center()))
        self.assertEqual(w.selected,['track:A'])
        self.assertNotIn('reference/g0',w.view_data['window']['mapping'])
        self.assertEqual(w.history[2].data,original)
        self.assertEqual(view.texts['track:A'].toPlainText(),'P0')
        from unittest.mock import patch
        with patch.object(w.history[2].bev,'open_original') as inspect:
            QtTest.QTest.mouseDClick(view.viewport(),QtCore.Qt.LeftButton,pos=QtCore.QPoint(15,15))
            inspect.assert_called_once_with()
        self.assertFalse(w.history[0].context_bev.isInteractive())
        w.layout_choice.setCurrentIndex(1)
        w.resize(1280,800)
        self.app.processEvents()
        self.assertLessEqual(w.height(),850)
        self.assertTrue(all(card.divided.orientation()==QtCore.Qt.Horizontal for card in w.history))
        w.mode.setCurrentIndex(1)
        self.wait(lambda:w.view_data is not None)
        self.assertIsNone(w.history[2].context)
        self.assertIsNone(w.view_data['cloud3d'])

    def test_w_s_browse_without_samples_and_from_closed_combo(self):
        w=self.window
        w.mode.setFocus()
        QtTest.QTest.keyClick(w.mode,QtCore.Qt.Key_W)
        self.wait(lambda:w.view_data is not None and w.frame_index==0)
        QtTest.QTest.keyClick(w.mode,QtCore.Qt.Key_S)
        self.wait(lambda:w.view_data is not None and w.frame_index==20)
        self.assertEqual(w.event['t1'],{})
        w.goto(12);self.wait(lambda:w.view_data is not None and w.frame_index==12)
        w.add_sample();self.idle()
        w.goto(5);self.wait(lambda:w.view_data is not None and w.frame_index==5)
        w.people.setFocus();QtTest.QTest.keyClick(w.people,QtCore.Qt.Key_S)
        self.wait(lambda:w.view_data is not None and w.frame_index==12)
        self.assertEqual(set(w.event['t1']),{'source/12'})
        w.events.setCurrentIndex(-1)
        self.wait(lambda:w.view_data is not None and w.event is None)
        w.sources.setFocus();QtTest.QTest.keyClick(w.sources,QtCore.Qt.Key_S)
        self.wait(lambda:w.view_data is not None and w.frame_index==30)

    def test_native_wasd_navigates_from_lists_combos_and_numeric_editors(self):
        w=self.window
        w.activateWindow()
        self.app.processEvents()
        w.interval.setValue(.5)
        w.goto(10)
        self.wait(lambda:w.view_data is not None and w.frame_index==10)
        for control in (w.people,w.groups,w.mode,w.layout_choice,w.relation,w.interval,w.interval.lineEdit()):
            for key,expected in ((QtCore.Qt.Key_D,11),(QtCore.Qt.Key_A,10),
                                 (QtCore.Qt.Key_W,5),(QtCore.Qt.Key_S,10)):
                with self.subTest(control=type(control).__name__,key=key):
                    control.setFocus()
                    self.app.processEvents()
                    QtTest.QTest.keyClick(w.windowHandle(),key)
                    self.assertEqual(w.frame_index,expected)
                    self.wait(lambda:w.view_data is not None)
                    self.assertEqual(w.interval.value(),.5)
                    self.assertEqual(w.mode.currentIndex(),0)
                    self.assertEqual(w.layout_choice.currentIndex(),0)
                    self.assertEqual(w.event['t1'],{})
        w.interval.setFocus()
        w.interval.selectAll()
        for key in (QtCore.Qt.Key_2,QtCore.Qt.Key_Period,QtCore.Qt.Key_5):
            QtTest.QTest.keyClick(w.windowHandle(),key)
        self.assertEqual(w.interval.value(),2.5)
        QtTest.QTest.keyClick(w.windowHandle(),QtCore.Qt.Key_Up)
        self.assertEqual(w.interval.value(),3.5)
        self.assertEqual(w.event['t1'],{})

    def test_native_wasd_leaves_text_and_modal_dialog_edits_alone(self):
        w=self.window
        w.activateWindow()
        w.reviewer.setFocus()
        w.reviewer.clear()
        self.app.processEvents()
        for key in (QtCore.Qt.Key_W,QtCore.Qt.Key_A,QtCore.Qt.Key_S,QtCore.Qt.Key_D):
            QtTest.QTest.keyClick(w.windowHandle(),key)
        self.assertEqual(w.reviewer.text(),'wasd')
        self.assertEqual(w.frame_index,20)
        QtTest.QTest.keyClick(w.windowHandle(),QtCore.Qt.Key_Escape)
        QtTest.QTest.keyClick(w.windowHandle(),QtCore.Qt.Key_D)
        self.assertEqual(w.frame_index,21)
        self.wait(lambda:w.view_data is not None)
        dialog=QtWidgets.QDialog(w)
        dialog.setModal(True)
        field=QtWidgets.QLineEdit(dialog)
        QtWidgets.QVBoxLayout(dialog).addWidget(field)
        try:
            dialog.show()
            dialog.activateWindow()
            field.setFocus()
            self.app.processEvents()
            for key in (QtCore.Qt.Key_W,QtCore.Qt.Key_A,QtCore.Qt.Key_S,QtCore.Qt.Key_D):
                QtTest.QTest.keyClick(dialog.windowHandle(),key)
            self.assertEqual(field.text(),'wasd')
            self.assertEqual(w.frame_index,21)
        finally:
            dialog.close()

    def test_wasd_overrides_child_shortcuts_and_closes_plain_combo_popup(self):
        from PyQt5 import QtGui
        w=self.window
        w.activateWindow()
        w.people.setFocus()
        competing=QtWidgets.QShortcut(QtGui.QKeySequence('D'),w.people)
        triggered=[]
        competing.activated.connect(lambda:triggered.append(True))
        self.app.processEvents()
        QtTest.QTest.keyClick(w.windowHandle(),QtCore.Qt.Key_D)
        self.assertEqual(w.frame_index,21)
        self.assertEqual(triggered,[])
        self.wait(lambda:w.view_data is not None)
        before=w.relation.currentIndex()
        w.relation.showPopup()
        try:
            self.app.processEvents()
            popup=self.app.activePopupWidget()
            self.assertIsNotNone(popup)
            QtTest.QTest.keyClick(popup.windowHandle(),QtCore.Qt.Key_W)
            self.assertEqual(w.frame_index,1)
            self.assertIsNone(self.app.activePopupWidget())
            self.assertEqual(w.relation.currentIndex(),before)
            self.wait(lambda:w.view_data is not None)
            self.assertEqual(w.event['t1'],{})
        finally:
            w.relation.hidePopup()


if __name__ == '__main__':
    unittest.main()
