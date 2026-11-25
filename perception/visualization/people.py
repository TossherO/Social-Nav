"""Read-only person display shared by live ROS and recorded-result viewing."""
from bisect import bisect_right
from collections import deque
from copy import copy
from itertools import count
import json
import math
from pathlib import Path

import numpy as np

from perception.detection_tracking.inference.geometry import corners_xyz, validate_boxes, validate_transform


EDGES = [(i, (i//4)*4+(i+1)%4) for i in range(8)] + [(i, i+4) for i in range(4)]
_EDGE_POINTS = np.asarray(EDGES).ravel()


def render_rgb_associations(pixels,people,result):
    """Display-only overlay; raw pixels, detections and identities remain unchanged."""
    import cv2
    output=np.asarray(pixels).copy()
    labels=[]
    for person in people:
        record=result['tracks'].get(person['key'],{})
        if record.get('status')!='matched' or record.get('rgb_box') is None:continue
        x1,y1,x2,y2=np.rint(record['rgb_box']).astype(int)
        color=(0,210,255) if person.get('observed') else (255,168,64)
        cv2.rectangle(output,(x1,y1),(x2,y2),color,2)
        label='ID '+person['key'].rsplit(':',1)[-1]
        (width,height),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.5,1)
        left=max(0,min(x1,output.shape[1]-width-6));bottom=max(height+6,y1-5)
        for _ in range(len(labels)+1):
            rect=(left,bottom-height-5,left+width+5,bottom+2)
            if not any(min(rect[2],r[2])>max(rect[0],r[0]) and min(rect[3],r[3])>max(rect[1],r[1]) for r in labels):break
            bottom=max(height+6,bottom-height-9)
        labels.append(rect)
        cv2.rectangle(output,rect[:2],rect[2:],(18,27,38),-1)
        cv2.putText(output,label,(left+2,bottom-2),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,245,250),1,cv2.LINE_AA)
    return output


class PeopleScene:
    def __init__(self, history_s=3.0, max_history_points=32, *, stable_marker_ids=False):
        if not math.isfinite(history_s) or history_s <= 0 or max_history_points < 2:
            raise ValueError('Invalid person display history budget')
        self.history_s, self.max_points = history_s, max_history_points
        self.stable_marker_ids = stable_marker_ids
        self.clear()

    def clear(self):
        self.identity, self.last_stamp, self.history = None, None, {}
        self.track_slots = {}

    def update(self, result, sensor_only=False, *, surface=None):
        stamp, frame = float(result['stamp_s']), result['frame_id']
        if not math.isfinite(stamp) or stamp < 0 or not frame or not result['detection_frame_id']:
            raise ValueError('Invalid person observation time/frame')
        if result['health'] not in ('valid', 'degraded', 'invalid'):
            raise ValueError('Unknown person health')
        identity = (result['session_id'], result['sequence_id'], result['generation'], frame)
        if identity == self.identity and self.last_stamp is not None and stamp <= self.last_stamp:
            return None  # A delayed publication must not replace the latest scene.
        if identity != self.identity:
            self.clear()
        self.identity, self.last_stamp = identity, stamp
        localized = not sensor_only and result['health'] == 'valid' and result['world_from_lidar'] is not None
        transform = validate_transform(result['world_from_lidar']) if localized else None
        specs = []

        def add(namespace, kind, color, points=(), position=(0, 0, 0), text='', sensor=False, slot=0):
            spec = dict(namespace=namespace, kind=kind, color=color,
                points=np.asarray(points, dtype=float).reshape(-1, 3), position=np.asarray(position, dtype=float),
                text=text, frame_id=result['detection_frame_id'] if sensor else frame, stamp_s=stamp)
            if self.stable_marker_ids:
                spec['id'] = slot
            specs.append(spec)

        def wire(box, namespace, color, sensor=False, slot=0):
            corners = corners_xyz(box)
            if sensor and transform is not None:
                corners = corners @ transform[:3, :3].T + transform[:3, 3]
                sensor = False
            lift = 0.
            if localized and surface is not None:
                bottom = corners[:4]
                lift = float(np.max(surface.above(bottom)[:, 2] - bottom[:, 2]))
                corners[:, 2] += lift
            add(namespace, 'box', color, corners[_EDGE_POINTS], sensor=sensor, slot=slot)
            return lift

        detections = result['detections'] if result['health'] != 'invalid' else ()
        for slot, box in enumerate(validate_boxes(detections)):
            wire(box, 'detections', (1, .85, .30, .70), sensor=True, slot=slot)
        active = set()
        tracks = result['tracks'] if localized else ()
        if self.stable_marker_ids:
            keys = [(track['track_id'], track['lifecycle_generation']) for track in tracks]
            slots = {key: self.track_slots[key] for key in keys if key in self.track_slots}
            used = set(slots.values())
            # Keep existing identities in place and reuse vacated slots for new ones.
            available = (slot for slot in count() if slot not in used)
            for key in keys:
                if key not in slots:
                    slots[key] = next(available)
            self.track_slots = slots
        for track in tracks:
            box = validate_boxes([track['box']])[0]
            observed = track['state'] == 'observed'
            color = ((.33, 1, .54, 1) if track['confirmed'] else (.49, .68, 1, 1)) if observed else (1, .59, .32, 1)
            key = (track['track_id'], track['lifecycle_generation'])
            slot = self.track_slots.get(key, 0)
            lift = wire(box, 'tracks', color, slot=slot)
            center = box[:3]+[0, 0, lift]
            label = 'ID %s / g%s %s %.2f' % (key[0].rsplit(':', 1)[-1], key[1],
                'OBS' if observed and track['confirmed'] else 'TENT' if observed else 'PRED', box[7])
            add('labels', 'text', color, position=center+[0, 0, box[5]/2+.3], text=label, slot=slot)
            velocity = track['velocity_xyz_mps']
            if velocity is not None:
                velocity = np.asarray(velocity, dtype=float)
                if velocity.shape != (3,) or not np.isfinite(velocity).all():
                    raise ValueError('Invalid displayed velocity')
                if np.linalg.norm(velocity) > .01:
                    add('velocity', 'arrow', color, [center, center+velocity], slot=slot)
            if observed and track['confirmed']:
                active.add(key)
                history = self.history.setdefault(key, deque(maxlen=self.max_points))
                history.append((stamp, box[:3].copy()))
                while history and stamp-history[0][0] > self.history_s:
                    history.popleft()
                if len(history) > 1:
                    add('history', 'trail', color, [point for _, point in history], slot=slot)
        # Predicted, expired or degraded tracks never extend an observed trail.
        self.history = {key: value for key, value in self.history.items() if key in active}
        anchor = transform[:3, 3] if localized else np.zeros(3)
        add('status', 'text', (.85, .9, 1, 1) if localized else (1, .3, .2, 1),
            position=anchor+[0, 0, 2.5], text='%s | %s | %d detections, %d tracks' % (
                result['health'], result.get('pose_source', 'unknown'), len(result['detections']),
                len(result['tracks']) if localized else 0), sensor=not localized)
        return specs


def marker_array(specs, lifetime_s=0.0):
    """ROS adapter; DELETEALL makes every publication a complete visual snapshot."""
    from visualization_msgs.msg import Marker, MarkerArray
    messages = [Marker(action=Marker.DELETEALL)]
    messages.extend(_ros_marker(spec, index, lifetime_s) for index, spec in enumerate(specs))
    return MarkerArray(markers=messages)


def _ros_marker(spec, number, lifetime_s=0.0):
    import rospy
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker
    kinds = {'box': Marker.LINE_LIST, 'trail': Marker.LINE_STRIP,
             'text': Marker.TEXT_VIEW_FACING, 'arrow': Marker.ARROW, 'sphere': Marker.SPHERE}
    marker = Marker(ns=spec['namespace'], id=number, type=kinds[spec['kind']])
    marker.header.frame_id = spec['frame_id']
    marker.header.stamp = rospy.Time.from_sec(spec['stamp_s'])
    marker.pose.orientation.w = 1
    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, spec['position'])
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = spec['color']
    marker.points = [Point(*point) for point in np.asarray(spec['points'],dtype=float).tolist()]
    marker.text = spec['text']
    scale = spec.get('scale')
    if scale is None:
        width = .03 if spec['namespace'] == 'detections' else .065
        scale = (width, 0., 0.)
        if spec['kind'] == 'text':
            scale = (width, 0., .25)
        elif spec['kind'] == 'arrow':
            scale = (.05, .12, .18)
    marker.scale.x, marker.scale.y, marker.scale.z = scale
    marker.lifetime = rospy.Duration(lifetime_s)
    return marker


class MarkerUpdates:
    """Live marker state with explicit removals and periodic recovery from lost arrays."""
    def __init__(self):
        self.active = {}
        # Slots are reused. The set grows only with peak simultaneous detections
        # and tracks, independent of the number of retired track identities.
        self.slots = set()
        self.epoch = None
        self.next_snapshot = 0.
        self.last_change = -math.inf
        self.initialized = False
        self.reset = True

    def update(self, specs, *, reset=False):
        import rospy
        from std_msgs.msg import Header
        active, changed = {}, []
        for spec in specs:
            key = (spec['namespace'], spec['id'])
            prior, marker = self.active.get(key, (None, None))
            if (prior is None or any(prior[k] != spec[k] for k in ('kind', 'color', 'text')) or
                    prior.get('scale') != spec.get('scale') or
                    any(not np.array_equal(prior[k], spec[k]) for k in ('position', 'points'))):
                marker = _ros_marker(spec, spec['id'])
                changed.append(marker)
            elif (prior['frame_id'], prior['stamp_s']) != (spec['frame_id'], spec['stamp_s']):
                # The geometry is read-only, but a new measurement may need a
                # different TF lookup even if the coordinates did not move.
                marker = copy(marker)
                marker.header = Header(frame_id=spec['frame_id'], stamp=rospy.Time.from_sec(spec['stamp_s']))
                changed.append(marker)
            active[key] = (spec, marker)
        removed = self.active.keys() - active.keys()
        self.active = active
        self.slots.update(active)
        self.reset |= reset
        self.initialized = True
        return changed, removed

    def message(self, epoch, now, changed=(), removed=(), *, force=False):
        from visualization_msgs.msg import Marker, MarkerArray
        if not self.initialized:
            return None
        updated = bool(changed or removed)
        if updated:
            self.last_change = now
        # Attach recovery to the next live update when possible. A paused feed
        # still repairs dropped deletions, and a new peer always gets a snapshot.
        due = now >= self.next_snapshot and (updated or now-self.last_change >= .3)
        full = force or self.reset or self.epoch != epoch or due
        if full:
            changed = [marker for _, marker in self.active.values()]
            removed = self.slots - self.active.keys()
            self.epoch = epoch
            self.next_snapshot = now + 1.
        # Deletes have no measured geometry or transform dependency. RViz
        # executes them even when the empty frame fails its TF filter.
        commands = [Marker(action=Marker.DELETEALL)] if self.reset else [
            Marker(ns=ns, id=slot, action=Marker.DELETE) for ns, slot in sorted(removed)]
        self.reset = False
        commands.extend(changed)
        return MarkerArray(markers=commands) if commands else None


class RecordedPeople:
    """Seekable past-only display; no model, raw data export or localization output."""
    def __init__(self, path, config):
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        self.stamps = [float(row['stamp_s']) for row in self.rows]
        if any(not math.isfinite(t) for t in self.stamps) or any(b <= a for a, b in zip(self.stamps, self.stamps[1:])):
            raise ValueError('Person records have invalid/reversed observation times')
        self.config = config

    def row_at(self, stamp):
        index = bisect_right(self.stamps, stamp)-1
        if index < 0 or stamp-self.stamps[index] > 1e-6:
            return None
        return self.rows[index]

    def at(self, stamp, sensor_only=False):
        row = self.row_at(stamp)
        if row is None:
            return []
        index = bisect_right(self.stamps, stamp)-1
        scene = PeopleScene(self.config['history_s'], self.config['max_history_points'])
        begin = bisect_right(self.stamps, self.stamps[index]-self.config['history_s'])
        for row in self.rows[begin:index+1]:
            specs = scene.update(row, sensor_only=sensor_only)
        return specs


def place_specs(specs, source_frame, target_frame, transform):
    """Rigidly place recorded local-world geometry in its selected display frame."""
    transform = validate_transform(transform)
    placed = []
    for spec in specs:
        if spec['frame_id'] != source_frame:
            continue  # No assumed transform for an unlocalized sensor-frame detection.
        copy = dict(spec, frame_id=target_frame)
        if len(spec['points']):
            copy['points'] = spec['points'] @ transform[:3, :3].T + transform[:3, 3]
        else:
            copy['position'] = spec['position'] @ transform[:3, :3].T + transform[:3, 3]
        placed.append(copy)
    return placed


