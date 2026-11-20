"""Ordered navigation requirements and evidence; never dispatches tool calls."""
import copy
import math
import numpy as np
from navigation.control.tracking import load_config
from perception.environment.context import fingerprint
from ..tools.framework import GOAL_VALIDATOR, REGISTRATION_GOAL_VALIDATOR


def user_quoted(quote, messages):
    return bool(quote and quote.strip()) and any(quote in message for message in messages)


def next_unfinished_goal(goals):
    return next((goal for goal in goals if goal['status'] not in ('completed','skipped')), None)


def normalize_registration(goal):
    """Convert the advertised registration contract to the internal record.

    Legacy fixture and historical actions use the internal shape directly.
    This is a field mapping, not a repair of malformed model arguments.
    """
    if 'stage' not in goal:
        result = copy.deepcopy(goal)
        if result.get('kind') == 'region' and result.get('position_rule') == 'region_centroid':
            del result['position_rule']
        return result
    REGISTRATION_GOAL_VALIDATOR.validate(goal)
    stage=goal['stage']
    navigation=stage in ('point','region','reference','route','sequence')
    target=goal.get('target',{})
    if not navigation and target:
        raise ValueError('A skill stage cannot supply a navigation target')
    if stage=='sequence':
        if set(target)!={'checkpoints'}:
            raise ValueError('Sequence target requires only ordered checkpoints')
    elif stage=='point':
        if set(target) not in ({'id'},{'position'}):
            raise ValueError('Point target requires exactly one annotation ID or user-specified position')
    elif stage=='region':
        if set(target)!={'id'}:
            raise ValueError('Region target requires one annotated region ID')
    elif stage in ('reference','route'):
        if set(target)-{'id'}:
            raise ValueError('Reference or route target may contain only an annotation ID')
    result=dict(goal_id=goal['goal_id'],kind=stage if navigation else 'skill',
                target_id=target.get('id',''),description=goal['description'])
    if not navigation:result['stage_type']=stage
    for key in ('position','checkpoints'):
        if key in target:result[key]=copy.deepcopy(target[key])
    for key in ('floor_id','to_floor','monitor','execution_notes'):
        if key in goal:result[key]=copy.deepcopy(goal[key])
    conditions=goal.get('conditions',{})
    for key in ('required_yaw','facing','max_distance_m','robot_inside_region_id','wait_after'):
        if key in conditions:result[key]=copy.deepcopy(conditions[key])
    for item in conditions.get('visual',[]):
        key='semantic_conditions' if item['priority']=='required' else 'visual_preferences'
        result.setdefault(key,[]).append(item['description'])
        if len(result[key])>8:
            raise ValueError('At most eight visual conditions are supported per priority')
    return result


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


def goal_requires_heading(requirement):
    return (requirement.get('stage_type')=='seating' or 'required_yaw' in requirement or
            any(requirement.get(key) for key in ('facing','semantic_conditions','visual_preferences')))


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
    @staticmethod
    def monitor_for(requirement):
        if 'monitor' in requirement:
            return requirement['monitor']
        interval={'follow':15.,'elevator':5.}.get(requirement.get('stage_type'))
        return {} if interval is None else dict(interval_s=interval)

    def execution_plan(self):
        plans=[]
        for goal in self.goals:
            req=goal['requirement']
            stage=req.get('stage_type','navigation')
            if stage=='navigation':
                tools=['navigate_route' if req['kind'] in ('route','sequence') else 'navigate']
            elif stage=='elevator':
                tools=['record_elevator_observation','restart_map_localization']
            elif stage=='observation':
                tools=['observe_scene']
            elif stage=='follow':
                tools=['select_follow_target','start_local_task']
            else:
                tools=['start_local_task']
            tools.append('complete_subtask')
            plans.append(dict(req,stage_type=stage,execution_tools=tools,monitor=self.monitor_for(req)))
        return plans

    def __init__(self, catalog, *, floor_catalogs=None):
        self.catalog = catalog
        self.floor_catalogs = dict(floor_catalogs or {catalog.value.get('floor_id'):catalog})
        if self.floor_catalogs.get(catalog.value.get('floor_id')) is not catalog:
            raise ValueError('Current catalog must be registered under its floor identity')
        if any(floor!=item.value.get('floor_id') for floor,item in self.floor_catalogs.items()):
            raise ValueError('Floor catalog identity does not match its registration')
        self.goals = []
        self.settings = load_config()
        # Task-level interpretation is shared across execution modes. The
        # navigation controller keeps its own independent arrival tolerance.
        self.settings['goal_position_tolerance_m'] = .20

    def record(self, goals, *, user_messages=()):
        self.catalog.require_complete()
        if self.goals:
            raise ValueError('Task requirements already recorded; replace the instruction before changing them')
        if not isinstance(goals, list) or not 1 <= len(goals) <= 16:
            raise ValueError('Use one to sixteen ordered requirements')
        normalized=[]
        for index,goal in enumerate(goals):
            try:normalized.append(normalize_registration(goal))
            except (ValueError,KeyError,TypeError) as error:
                raise ValueError(f'goals[{index}]: {error}') from error
        ids = set()
        active_floor=self.catalog.value.get('floor_id')
        for index,goal in enumerate(normalized):
            try:
                active_floor=self._validate_requirement(goal,active_floor,ids)
                if 'wait_after' in goal and not user_quoted(goal['wait_after'],user_messages):
                    raise ValueError('wait_after must quote an explicit wait instruction from the user')
            except ValueError as error:
                raise ValueError(f'goals[{index}]: {error}') from error
        self.goals = [dict(requirement=copy.deepcopy(g), binding=fingerprint(g), status='pending',
                           selected=None, completion=None) for g in normalized]
        return self.snapshot()

    def _validate_requirement(self,goal,active_floor,ids):
        GOAL_VALIDATOR.validate(goal)
        if goal['goal_id'] in ids:
            raise ValueError('Duplicate task goal identity')
        ids.add(goal['goal_id'])
        if len(self.floor_catalogs)>1 and 'floor_id' not in goal:
            raise ValueError('Cross-floor task stages require explicit floor identities')
        floor=goal.get('floor_id',active_floor)
        if floor!=active_floor or floor not in self.floor_catalogs:
            raise ValueError('Task stage must follow the current floor and known transitions')
        catalog=self.floor_catalogs[floor]
        catalog.require_complete()
        regions = goal.get('monitor',{}).get('region_ids',[])
        if len(set(regions)) != len(regions) or any(region not in catalog.regions for region in regions):
            raise ValueError(f"Goal {goal['goal_id']} monitor.region_ids must identify distinct regions on its floor")
        monitor=goal.get('monitor',{})
        if ('continuous_region_id' in monitor) != ('review_mode' in monitor):
            raise ValueError('Continuous regional review requires both continuous_region_id and review_mode')
        if monitor.get('continuous_region_id') is not None and monitor['continuous_region_id'] not in catalog.regions:
            raise ValueError('Continuous review requires a region on the current floor')
        if goal.get('stage_type')=='elevator' and set(goal.get('monitor',{}))-{'focus','interval_s'}:
            raise ValueError('Elevator monitoring supports independent frame events and interval_s while localization is paused')
        target = goal['target_id']
        if goal['kind'] != 'skill' and goal.get('stage_type', 'navigation') != 'navigation':
            raise ValueError('Non-navigation stage types require kind=skill')
        if goal['kind'] == 'skill':
            if (target or goal.get('stage_type') not in ('follow', 'observation', 'seating','elevator') or
                    any(key in goal for key in ('position', 'required_yaw', 'facing',
                                                'max_distance_m', 'checkpoints'))):
                raise ValueError('A skill stage requires its stage_type and an empty target, without navigation geometry')
            if goal.get('stage_type')=='elevator':
                destination=goal.get('to_floor')
                if len(self.floor_catalogs)<2 or destination not in self.floor_catalogs or destination==floor:
                    raise ValueError('Elevator transition requires a different registered destination floor')
                if goal.get('semantic_conditions') or goal.get('visual_preferences'):
                    raise ValueError('Record additional visual requirements as a separate task stage')
            elif 'to_floor' in goal:
                raise ValueError('Only an elevator stage may name a destination floor')
        elif goal['kind'] == 'region':
            if target not in catalog.regions or 'position' in goal:
                raise ValueError('Region requirements use an annotated region, not a selected position')
        elif goal['kind'] == 'route':
            if 'circuit' in goal and target not in catalog.regions:
                raise ValueError('A circuit needs the annotated region it surrounds')
            if 'position' in goal or target and target not in catalog.regions and target not in catalog.places:
                raise ValueError('Route reference must be a known annotation or empty')
        elif goal['kind'] == 'sequence':
            if target or 'position' in goal or not goal.get('checkpoints'):
                raise ValueError('Sequence uses ordered annotated checkpoints and an empty target')
            for checkpoint in goal['checkpoints']:
                if checkpoint not in catalog.regions and (checkpoint not in catalog.places or
                        catalog.places[checkpoint].get('pose') is None):
                    raise ValueError('Sequence checkpoint is not bound in this map')
        elif goal['kind'] == 'reference':
            if 'position' in goal:
                raise ValueError('Reference requirements do not fix the selected position')
            if target and target not in catalog.regions and (target not in catalog.places or
                    catalog.places[target].get('pose') is None):
                raise ValueError('Reference is not bound in this map')
            if not target and not goal.get('semantic_conditions') and not goal.get('visual_preferences'):
                raise ValueError('An unnamed reference needs an observable original task condition or visual preference')
            if 'max_distance_m' in goal and not target:
                raise ValueError('A geometric distance needs an annotated reference')
        elif target:
            if target not in catalog.places or catalog.places[target].get('pose') is None:
                raise ValueError('Point is not bound in this map')
            if 'position' in goal and not np.allclose(goal['position'], catalog.places[target]['pose'][:2], atol=1e-9):
                raise ValueError('Explicit position contradicts the named point')
        elif 'position' not in goal:
            raise ValueError('An unnamed point requires a user-specified position')
        if 'position_rule' in goal:
            raise ValueError('The legacy position rule only applies to recorded region goals')
        if goal.get('checkpoints') and goal['kind'] != 'sequence':
            raise ValueError('Checkpoints apply only to sequences')
        if 'max_distance_m' in goal and goal['kind'] != 'reference':
            raise ValueError('Distance applies only to references')
        if goal.get('facing') and goal['facing'] not in catalog.regions and (goal['facing'] not in catalog.places or
                                   catalog.places[goal['facing']].get('pose') is None):
            raise ValueError('Facing condition needs a bound annotated point or region')
        if goal['kind']!='skill' and 'to_floor' in goal:
            raise ValueError('Only an elevator stage may name a destination floor')
        completion_floor = goal.get('to_floor', floor)
        region_id = goal.get('robot_inside_region_id')
        if region_id is not None and region_id not in self.floor_catalogs[completion_floor].regions:
            raise ValueError(f'Completion region {region_id!r} does not exist on floor {completion_floor!r}')
        values = [*goal.get('position', []), *([goal['required_yaw']] if 'required_yaw' in goal else []),
                  *([goal['max_distance_m']] if 'max_distance_m' in goal else [])]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Requirements must be finite')
        return completion_floor

    def current(self, goal_id):
        current = next_unfinished_goal(self.goals)
        if current is None or current['requirement']['goal_id'] != goal_id:
            raise ValueError('Only the next uncompleted task goal can be selected or checked')
        floor=current['requirement'].get('floor_id')
        if current.get('destination_localization'):
            floor=current['requirement']['to_floor']
        if floor is not None and floor!=self.catalog.value.get('floor_id'):
            raise ValueError('Current task stage belongs to another floor')
        return current

    def geometry(self, requirement, pose):
        if np.shape(pose) not in ((2,),(3,)) or not np.isfinite(pose).all():
            raise ValueError('Expected a finite base_link XY or XY/yaw goal')
        if requirement['kind'] == 'point':
            point = requirement.get('position')
            if point is None:
                point = self.catalog.places[requirement['target_id']]['pose'][:2]
            position_met = float(np.linalg.norm(np.asarray(pose[:2])-point)) <= self.settings['goal_position_tolerance_m']
        elif requirement['kind']=='region':
            position_met = inside(pose[:2], self.catalog.regions[requirement['target_id']]['polygon'])
        elif requirement['kind']=='route':
            position_met = True
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
        yaw_met = ('required_yaw' not in requirement or
                   (len(pose)==3 and angle_error(pose[2],requirement['required_yaw'])<=self.settings['goal_yaw_tolerance_rad']))
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
        if current['requirement'].get('stage_type', 'navigation') != 'navigation':
            return dict(status='refused', reason='stage_skill_not_available',
                        stage_type=current['requirement']['stage_type'])
        checks = self.geometry(current['requirement'], pose)
        if any(value is False for value in checks.values()):
            return dict(status='refused', reason='selected_pose_violates_task_requirement', checks=checks)
        allowed = self.orientation_sources(current['requirement'])
        if len(pose)==3 and orientation_source not in allowed:
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

    def skip(self, goal_id):
        current = self.current(goal_id)
        current.update(status='skipped', skip=dict(source='operator', reason='operator_skipped_subtask'))

    def snapshot(self):
        return copy.deepcopy(self.goals)

    def finished(self):
        return bool(self.goals) and all(g['status'] == 'completed' for g in self.goals)

    def resolved(self):
        return bool(self.goals) and all(g['status'] in ('completed','skipped') for g in self.goals)
