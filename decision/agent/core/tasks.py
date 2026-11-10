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


def centroid(polygon):
    vertices = np.asarray(polygon,dtype=float)
    following = np.roll(vertices,-1,axis=0)
    cross = vertices[:,0]*following[:,1]-following[:,0]*vertices[:,1]
    area6 = 3*cross.sum()
    if abs(area6) < 1e-10:
        raise ValueError('Region has no area centroid')
    return ((vertices+following)*cross[:,None]).sum(axis=0)/area6


def distance_to_region(point, polygon):
    point = np.asarray(point,dtype=float)
    if inside(point,polygon):
        return 0.
    vertices = np.asarray(polygon,dtype=float)
    best = float('inf')
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        edge=b-a; length=float(edge@edge)
        t=0. if length == 0 else float(np.clip((point-a)@edge/length,0.,1.))
        best=min(best,float(np.linalg.norm(point-(a+t*edge))))
    return best


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
                if goal.get('position_rule') == 'region_centroid' and not inside(centroid(self.catalog.regions[target]['polygon']),self.catalog.regions[target]['polygon']):
                    raise ValueError('Region area centroid lies outside the region; clarify the intended center')
            elif goal['kind'] == 'circuit':
                if target not in self.catalog.regions or 'position' in goal:
                    raise ValueError('Circuit requires a region')
            elif goal['kind'] == 'sequence':
                if target or 'position' in goal or not goal.get('checkpoints'):
                    raise ValueError('Sequence uses ordered annotated checkpoints and an empty target')
                for checkpoint in goal['checkpoints']:
                    if checkpoint not in self.catalog.regions and (checkpoint not in self.catalog.places or
                            self.catalog.places[checkpoint].get('pose') is None):
                        raise ValueError('Sequence checkpoint is not bound in this map')
            elif goal['kind'] == 'reference':
                if 'position' in goal:
                    raise ValueError('Reference requirements do not fix the selected position')
                if target and target not in self.catalog.regions and (target not in self.catalog.places or
                        self.catalog.places[target].get('pose') is None):
                    raise ValueError('Reference is not bound in this map')
                if not goal.get('semantic_conditions') and 'max_distance_m' not in goal:
                    raise ValueError('Reference needs a geometric distance or an observable original task condition')
                if 'max_distance_m' in goal and not target:
                    raise ValueError('A geometric distance needs an annotated reference')
            elif target:
                if target not in self.catalog.places or self.catalog.places[target].get('pose') is None:
                    raise ValueError('Point is not bound in this map')
                if 'position' in goal and not np.allclose(goal['position'], self.catalog.places[target]['pose'][:2], atol=1e-9):
                    raise ValueError('Explicit position contradicts the named point')
            elif 'position' not in goal:
                raise ValueError('An unnamed point requires a user-specified position')
            if goal.get('position_rule') and goal['kind'] != 'region':
                raise ValueError('Position rule applies only to regions')
            if goal.get('direction') and goal['kind'] != 'circuit':
                raise ValueError('Direction applies only to circuits')
            if goal.get('checkpoints') and goal['kind'] != 'sequence':
                raise ValueError('Checkpoints apply only to sequences')
            if 'max_distance_m' in goal and goal['kind'] != 'reference':
                raise ValueError('Distance applies only to references')
            if goal.get('facing') and goal['facing'] not in self.catalog.regions and (goal['facing'] not in self.catalog.places or
                                       self.catalog.places[goal['facing']].get('pose') is None):
                raise ValueError('Facing condition needs a bound annotated point or region')
            values = [*goal.get('position', []), *([goal['required_yaw']] if 'required_yaw' in goal else []),
                      *([goal['max_distance_m']] if 'max_distance_m' in goal else [])]
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
        elif requirement['kind']=='region':
            position_met = inside(pose[:2], self.catalog.regions[requirement['target_id']]['polygon'])
            if requirement.get('position_rule') == 'region_centroid':
                position_met = position_met and np.linalg.norm(np.asarray(pose[:2])-centroid(
                    self.catalog.regions[requirement['target_id']]['polygon'])) <= self.settings['goal_position_tolerance_m']
        elif requirement['kind']=='circuit':
            position_met = not inside(pose[:2],self.catalog.regions[requirement['target_id']]['polygon'])
        elif requirement['kind']=='sequence':
            position_met = self.checkpoint_met(requirement['checkpoints'][-1],pose[:2])
        else:
            # This requirement supplies semantic context, not an implicit radius
            # or an assertion that the chosen coordinate satisfies the task.
            position_met = True
            if 'max_distance_m' in requirement:
                target = requirement['target_id']
                distance = (distance_to_region(pose[:2],self.catalog.regions[target]['polygon']) if target in self.catalog.regions else
                            float(np.linalg.norm(np.asarray(pose[:2])-self.catalog.places[target]['pose'][:2])))
                position_met = distance <= requirement['max_distance_m']
        yaw_met = 'required_yaw' not in requirement or angle_error(pose[2], requirement['required_yaw']) <= self.settings['goal_yaw_tolerance_rad']
        # A named object or area can be behind a wall. Its center alone does
        # not define the direction the user means at this particular pose.
        facing_met = None if requirement.get('facing') else True
        return dict(position_met=bool(position_met), required_yaw_met=bool(yaw_met), facing_met=facing_met)

    def checkpoint_met(self, checkpoint, point):
        if checkpoint in self.catalog.regions:
            return inside(point,self.catalog.regions[checkpoint]['polygon'])
        return float(np.linalg.norm(np.asarray(point)-self.catalog.places[checkpoint]['pose'][:2])) <= self.settings['goal_position_tolerance_m']

    def select(self, goal_id, pose, orientation_source, view_id):
        self.catalog.require_complete()
        current = self.current(goal_id)
        checks = self.geometry(current['requirement'], pose)
        if any(value is False for value in checks.values()):
            return dict(status='refused', reason='selected_pose_violates_task_requirement', checks=checks)
        allowed = self.orientation_sources(current['requirement'])
        if orientation_source not in allowed:
            return dict(status='refused',reason='orientation_source_disagrees_with_registered_requirement',
                        allowed_orientation_sources=allowed)
        selected = dict(goal_id=goal_id, task_goal_binding=current['binding'],
                        place_id=current['requirement']['target_id'] or goal_id, pose=list(pose),
                        reference='base_link', frame_id=self.catalog.value['frame_id'],
                        orientation_source=orientation_source, view_id=view_id,
                        requirement=copy.deepcopy(current['requirement']))
        current['selected'] = copy.deepcopy(selected)
        return dict(status='selected', destination=selected)

    @staticmethod
    def orientation_sources(requirement):
        return ['user'] if 'required_yaw' in requirement else ['semantic','autonomous']

    def complete(self, goal_id, proof, checks, semantic):
        current = self.current(goal_id)
        current.update(status='completed', completion=dict(proof=copy.deepcopy(proof),
                       checks=copy.deepcopy(checks), semantic=copy.deepcopy(semantic)))

    def snapshot(self):
        return copy.deepcopy(self.goals)

    def finished(self):
        return bool(self.goals) and all(g['status'] == 'completed' for g in self.goals)
