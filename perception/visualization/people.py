"""Read-only person display shared by live ROS and recorded-result viewing."""
from bisect import bisect_right
from collections import deque
import json
import math
from pathlib import Path

import numpy as np

from perception.detection_tracking.inference.geometry import corners_xyz, validate_boxes, validate_transform


EDGES = [(i, (i//4)*4+(i+1)%4) for i in range(8)] + [(i, i+4) for i in range(4)]


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
    def __init__(self, history_s=3.0, max_history_points=32):
        if not math.isfinite(history_s) or history_s <= 0 or max_history_points < 2:
            raise ValueError('Invalid person display history budget')
        self.history_s, self.max_points = history_s, max_history_points
        self.clear()

    def clear(self):
        self.identity, self.last_stamp, self.history = None, None, {}

    def update(self, result, sensor_only=False):
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

        def add(namespace, kind, color, points=(), position=(0, 0, 0), text='', sensor=False):
            specs.append(dict(namespace=namespace, kind=kind, color=color,
                points=np.asarray(points, dtype=float).reshape(-1, 3), position=np.asarray(position, dtype=float),
                text=text, frame_id=result['detection_frame_id'] if sensor else frame, stamp_s=stamp))

        def wire(box, namespace, color, sensor=False):
            corners = corners_xyz(validate_boxes([box])[0])
            if sensor and transform is not None:
                corners = corners @ transform[:3, :3].T + transform[:3, 3]
                sensor = False
            add(namespace, 'box', color, [corners[i] for edge in EDGES for i in edge], sensor=sensor)

        for box in result['detections'] if result['health'] != 'invalid' else ():
            wire(box, 'detections', (1, .8, .1, .65), sensor=True)
        active = set()
        for track in result['tracks'] if localized else ():
            box = validate_boxes([track['box']])[0]
            observed = track['state'] == 'observed'
            color = ((.2, 1, .4, 1) if track['confirmed'] else (.3, .65, 1, 1)) if observed else (1, .5, .1, 1)
            wire(box, 'tracks', color)
            key = (track['track_id'], track['lifecycle_generation'])
            label = 'ID %s / g%s %s %.2f' % (key[0].rsplit(':', 1)[-1], key[1],
                'OBS' if observed and track['confirmed'] else 'TENT' if observed else 'PRED', box[7])
            add('labels', 'text', color, position=box[:3]+[0, 0, box[5]/2+.3], text=label)
            velocity = track['velocity_xyz_mps']
            if velocity is not None:
                velocity = np.asarray(velocity, dtype=float)
                if velocity.shape != (3,) or not np.isfinite(velocity).all():
                    raise ValueError('Invalid displayed velocity')
                if np.linalg.norm(velocity) > .01:
                    add('velocity', 'arrow', color, [box[:3], box[:3]+velocity])
            if observed and track['confirmed']:
                active.add(key)
                history = self.history.setdefault(key, deque(maxlen=self.max_points))
                history.append((stamp, box[:3].copy()))
                while history and stamp-history[0][0] > self.history_s:
                    history.popleft()
                if len(history) > 1:
                    add('history', 'trail', color, [point for _, point in history])
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
    import rospy
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker, MarkerArray
    reset = Marker(action=Marker.DELETEALL)
    messages = [reset]
    kinds = {'box': Marker.LINE_LIST, 'trail': Marker.LINE_STRIP,
             'text': Marker.TEXT_VIEW_FACING, 'arrow': Marker.ARROW}
    for index, spec in enumerate(specs):
        marker = Marker()
        marker.header.frame_id = spec['frame_id']
        marker.header.stamp = rospy.Time.from_sec(spec['stamp_s'])
        marker.ns, marker.id, marker.type = spec['namespace'], index, kinds[spec['kind']]
        marker.pose.orientation.w = 1
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = spec['position']
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = spec['color']
        marker.points = [Point(*point) for point in spec['points']]
        marker.text = spec['text']
        marker.scale.x = .025 if spec['namespace'] == 'detections' else .05
        if spec['kind'] == 'text':
            marker.scale.z = .25
        elif spec['kind'] == 'arrow':
            marker.scale.x, marker.scale.y, marker.scale.z = .05, .12, .18
        marker.lifetime = rospy.Duration(lifetime_s)
        messages.append(marker)
    return MarkerArray(markers=messages)


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


