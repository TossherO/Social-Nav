"""Conservative execution-only enclosure for uncertain vehicle responses."""
import math
import numpy as np
from navigation.planning.footprint import VehicleGeometry


def _rotation_footprint(footprint, angular_travel):
    """Axis-aligned body-frame box containing the footprint at any allowed yaw."""
    xs, ys = [], []
    for x, y in footprint:
        angles = [-angular_travel, 0., angular_travel]
        # x(phi)=x*cos(phi)-y*sin(phi); y(phi)=x*sin(phi)+y*cos(phi).
        for extremum in (math.atan2(-y, x), math.atan2(x, y)):
            for k in range(-2, 3):
                angle = extremum + k * math.pi
                if -angular_travel <= angle <= angular_travel:
                    angles.append(angle)
        for angle in angles:
            c, s = math.cos(angle), math.sin(angle)
            xs.append(x*c-y*s);ys.append(x*s+y*c)
    return [[min(xs),min(ys)],[max(xs),min(ys)],[max(xs),max(ys)],[min(xs),max(ys)]]


def validate_response_bounds(value, motion):
    """Require physical limits, not command limits or a single test result."""
    names = {'max_planar_speed_mps', 'max_stop_time_s'}
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError('Execution requires measured planar-speed and stop-time bounds')
    for name in names:
        v = value[name]
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError('Invalid execution response bound: ' + name)
    if value['max_planar_speed_mps'] < max(motion['max_forward_mps'], motion['max_reverse_mps']):
        raise ValueError('Planar-speed bound is below selected command range')
    return dict(value)


def validate_execution_motion(motion):
    bounds = validate_response_bounds(motion.get('execution_response_bounds'), motion)
    angular = motion.get('execution_max_angular_rps')
    if (type(angular) not in (int, float) or not math.isfinite(angular) or
            angular < motion['max_angular_rps']):
        raise ValueError('Execution requires a measured angular-speed bound covering commands')
    return bounds, angular


def execution_response_enclosure(feedback, geometry, motion, settings, dt, observation_age_s):
    """Return a time and isotropic footprint expansion for every bounded path.

    Assumptions are explicit: physical planar speed and absolute yaw rate are
    always within the declared bounds, and the chair is stationary no later
    than max_stop_time_s after the last nonzero command or a zero command.
    Stop time includes controller response and braking after the final nonzero
    command. Duration adds source age, reaction and one control period; adding
    actuator delay again would double-count it. The expansion is independent
    of control direction. Rotations use an exact body-frame bounding rectangle
    over every yaw in the interval, rather than inflating all sides by the
    maximum corner displacement.
    """
    bounds, angular = validate_execution_motion(motion)
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(observation_age_s) or observation_age_s < 0:
        raise ValueError('Invalid response-enclosure timing')
    duration = observation_age_s + settings['reaction_s'] + dt + bounds['max_stop_time_s']
    translation = bounds['max_planar_speed_mps'] * duration
    angular_travel = min(math.pi, angular * duration)
    rotation_margin = 2. * geometry.radius * math.sin(angular_travel / 2.)
    if not math.isfinite(duration) or not math.isfinite(translation):
        raise ValueError('Unbounded execution response enclosure')
    pose = np.asarray(feedback['pose'], float)
    return dict(duration_s=duration, translation_bound_m=translation,
        rotation_bound_rad=angular_travel, rotation_margin_m=rotation_margin,
        additional_margin_m=translation,
        footprint_xy_m=_rotation_footprint(geometry.footprint,angular_travel),
        max_planar_speed_mps=bounds['max_planar_speed_mps'],
        max_yaw_rate_rps=angular, max_stop_time_s=bounds['max_stop_time_s'],
        points=np.array([[0., *pose], [duration, *pose]], dtype=float))


def execution_enclosure_geometry(geometry, enclosure):
    """Preserve vehicle reference frame while enclosing the entire yaw range."""
    return VehicleGeometry(enclosure['footprint_xy_m'], geometry.base_from_lidar,
        source=geometry.source, assumed=geometry.assumed)
