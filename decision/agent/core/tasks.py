"""Ordered navigation requirements and evidence; never dispatches tool calls."""
import copy
import math
import numpy as np
import jsonschema
from navigation.control.tracking import load_config
from perception.environment.context import fingerprint
from ..tools.framework import GOAL


def inside(point, polygon):
    """Boundary-inclusive point in a simple, not necessarily convex polygon."""
    p = np.asarray(point, float)
    v = np.asarray(polygon, float)
    crossed = False
    for a, b in zip(v, np.roll(v, -1, axis=0)):
        edge, delta = b-a, p-a
        if abs(edge[0]*delta[1]-edge[1]*delta[0]) <= 1e-9 and np.dot(delta, p-b) <= 1e-9:
            return True
        if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < a[0]+(p[1]-a[1])*edge[0]/edge[1]:
            crossed = not crossed
    return crossed


def angle_error(a, b):
    return abs(math.atan2(math.sin(a-b), math.cos(a-b)))


class TaskGoals:
    def __init__(self, catalog):
        self.catalog = catalog
        self.goals = []
        self.settings = load_config()

    def record(self, goals):
        self.catalog.require_complete()
        if self.goals:
            raise ValueError('Task requirements already recorded; replace the instruction before changing them')
        if not isinstance(goals, list) or not 1 <= len(goals) <= 16:
            raise ValueError('Use one to sixteen ordered requirements')
        ids = set()
        for goal in goals:
            jsonschema.validate(goal, GOAL)
            if goal['goal_id'] in ids:
                raise ValueError('Duplicate task goal identity')
            ids.add(goal['goal_id'])
            target = goal['target_id']
            if goal['kind'] == 'region':
                if target not in self.catalog.regions or 'position' in goal:
                    raise ValueError('Region requirements use an annotated region, not a selected position')
            elif target:
                if target not in self.catalog.places or self.catalog.places[target].get('pose') is None:
                    raise ValueError('Point is not bound in this map')
                if 'position' in goal and not np.allclose(goal['position'], self.catalog.places[target]['pose'][:2], atol=1e-9):
                    raise ValueError('Explicit position contradicts the named point')
            elif 'position' not in goal:
                raise ValueError('An unnamed point requires a user-specified position')
            if goal.get('facing') and goal['facing'] not in self.catalog.landmarks:
                raise ValueError('Facing condition needs an annotated landmark')
            values = [*goal.get('position', []), *([goal['required_yaw']] if 'required_yaw' in goal else [])]
            if not all(math.isfinite(v) for v in values):
                raise ValueError('Requirements must be finite')
        self.goals = [dict(requirement=copy.deepcopy(g), binding=fingerprint(g), status='pending',
                           selected=None, completion=None) for g in goals]
        return self.snapshot()

    def current(self, goal_id):
        current = next((g for g in self.goals if g['status'] != 'completed'), None)
        if current is None or current['requirement']['goal_id'] != goal_id:
            raise ValueError('Only the next uncompleted task goal can be selected or checked')
        return current

    def geometry(self, requirement, pose):
        if np.shape(pose) != (3,) or not np.isfinite(pose).all():
            raise ValueError('Expected a finite base_link XY/yaw pose')
        if requirement['kind'] == 'point':
            point = requirement.get('position')
            if point is None:
                point = self.catalog.places[requirement['target_id']]['pose'][:2]
            position_met = float(np.linalg.norm(np.asarray(pose[:2])-point)) <= self.settings['goal_position_tolerance_m']
        else:
            position_met = inside(pose[:2], self.catalog.regions[requirement['target_id']]['polygon'])
        yaw_met = 'required_yaw' not in requirement or angle_error(pose[2], requirement['required_yaw']) <= self.settings['goal_yaw_tolerance_rad']
        facing_met = True
        if requirement.get('facing'):
            point = self.catalog.landmarks[requirement['facing']]['position']
            delta = np.asarray(point)-pose[:2]
            facing_met = np.linalg.norm(delta) > 1e-6 and angle_error(pose[2], math.atan2(delta[1], delta[0])) <= self.settings['goal_yaw_tolerance_rad']
        return dict(position_met=bool(position_met), required_yaw_met=bool(yaw_met), facing_met=bool(facing_met))

    def select(self, goal_id, pose, orientation_source, view_id):
        self.catalog.require_complete()
        current = self.current(goal_id)
        checks = self.geometry(current['requirement'], pose)
        if not all(checks.values()):
            return dict(status='refused', reason='selected_pose_violates_task_requirement', checks=checks)
        selected = dict(goal_id=goal_id, task_goal_binding=current['binding'],
                        place_id=current['requirement']['target_id'] or goal_id, pose=list(pose),
                        reference='base_link', frame_id=self.catalog.value['frame_id'],
                        orientation_source=orientation_source, view_id=view_id,
                        requirement=copy.deepcopy(current['requirement']))
        current['selected'] = copy.deepcopy(selected)
        return dict(status='selected', destination=selected)

    def complete(self, goal_id, proof, checks, semantic):
        current = self.current(goal_id)
        current.update(status='completed', completion=dict(proof=copy.deepcopy(proof),
                       checks=copy.deepcopy(checks), semantic=copy.deepcopy(semantic)))

    def snapshot(self):
        return copy.deepcopy(self.goals)

    def finished(self):
        return bool(self.goals) and all(g['status'] == 'completed' for g in self.goals)
