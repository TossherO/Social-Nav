"""Opt-in single authority for continuous pose and separately timed map correction."""
import copy
import hashlib
import json
import threading
from pathlib import Path

from perception.mapping_localization.context import LocalizationContext
from perception.mapping_localization.results import map_fingerprint


def state_row(message):
    p, q = message.pose.position, message.pose.orientation
    mp, mq = message.map_pose.position, message.map_pose.orientation
    return dict(stamp_s=message.header.stamp.to_sec(), frame_id=message.header.frame_id,
                session_id=message.session_id, generation=message.generation, source=message.source,
                health=message.health, reason=message.reason, position=[p.x, p.y, p.z],
                quaternion_xyzw=[q.x, q.y, q.z, q.w], map_position=[mp.x, mp.y, mp.z],
                map_quaternion_xyzw=[mq.x, mq.y, mq.z, mq.w], map_frame=message.map_frame,
                map_revision=message.map_revision, has_map_pose=message.has_map_pose)


def association_row(message):
    from perception.mapping_localization.context import pose_matrix
    result = dict(stamp_s=message.header.stamp.to_sec(),stamp_ns=message.header.stamp.to_nsec(),local_frame=message.header.frame_id,
        local_session_id=message.session_id,local_generation=message.generation,
        map_id=message.map_id,asset_fingerprint=message.asset_fingerprint,map_frame=message.map_frame,
        association_session=message.association_session,association_generation=message.association_generation,
        map_revision=message.map_revision,valid=message.valid,status=message.status,reason=message.reason,
        measurement_stamp_s=message.measurement_stamp.to_sec() if message.has_measurement_stamp else None,
        age_s=message.age_s if message.has_measurement_stamp else None,
        coverage_status=message.coverage_status,coverage_fingerprint=message.coverage_fingerprint or None,
        map_from_local=None)
    if message.valid:
        p,q = message.map_from_local.translation,message.map_from_local.rotation
        result['map_from_local'] = pose_matrix([p.x,p.y,p.z],[q.x,q.y,q.z,q.w]).tolist()
    return result


def main():
    import rospy
    from geometry_msgs.msg import TransformStamped
    from tf.transformations import quaternion_from_matrix
    from tf2_ros import TransformBroadcaster
    from social_nav_localization.msg import LocalizationState, MapAssociation
    from nav_msgs.msg import Path as PosePath
    from geometry_msgs.msg import PoseStamped
    rospy.init_node('localization_context')
    import yaml
    config = yaml.safe_load(Path(rospy.get_param('~config')).read_text())
    mode = rospy.get_param('~mode')
    source = ('fast_lio' if mode == 'odometry' else 'roll_fast_lio')
    if rospy.get_param('/stationary_imu/enabled', False):
        source += '_stationary_assumption'
    map_path = rospy.get_param('~map_dir', '')
    fingerprint = ''
    if mode == 'localization':
        fingerprint = hashlib.sha256(json.dumps(map_fingerprint(map_path), sort_keys=True).encode()).hexdigest()
    map_id = 'roll:'+fingerprint if fingerprint else ''
    coverage = None
    coverage_path = rospy.get_param('~coverage','')
    if coverage_path:
        from perception.mapping_localization.coverage import PriorCoverage
        coverage = PriorCoverage.load(coverage_path,asset_fingerprint=fingerprint)
    context = LocalizationContext(source, requires_map=mode == 'localization',
        allow_local_only=rospy.get_param('~allow_local_only', False),
        max_map_age=config['association_max_age_s'],
        confirm_matches=config['association_confirm_matches'],coverage=coverage)
    event_path = rospy.get_param('~event_log','')
    events = Path(event_path).open('x') if event_path else None
    state_pub = rospy.Publisher('/localization/state', LocalizationState, queue_size=8)
    map_pub = rospy.Publisher('/localization/map_association', MapAssociation, queue_size=8)
    path_pub = rospy.Publisher('/localization/path', PosePath, queue_size=1)
    local_path = PosePath()
    path_identity = [None]
    transforms = TransformBroadcaster()
    lock = threading.RLock()
    if events:
        def close_events():
            with lock:
                events.close()
        rospy.on_shutdown(close_events)

    def transform(matrix, parent, child, stamp):
        result = TransformStamped()
        result.header.stamp, result.header.frame_id, result.child_frame_id = stamp, parent, child
        result.transform.translation.x, result.transform.translation.y, result.transform.translation.z = matrix[:3, 3]
        q = quaternion_from_matrix(matrix)
        for key, value in zip(('x','y','z','w'), q):
            setattr(result.transform.rotation, key, value)
        return result

    def match(message):
        with lock:
            row = state_row(message)
            if events and not events.closed:
                events.write(json.dumps(dict(kind='match',row=row))+'\n')
            context.add_match(row)

    def frontend(message):
        with lock:
            if events and not events.closed:
                events.write(json.dumps(dict(kind='frontend',row=state_row(message)))+'\n')
            pair = context.update(state_row(message))
            if pair is None:
                return
            row, association = pair
            output = copy.deepcopy(message)
            for key in ('source', 'health', 'status', 'reason', 'has_map_pose', 'map_frame', 'map_revision'):
                setattr(output, key, row[key])
            if row['has_map_pose']:
                matrix = row['map_matrix']
                output.map_pose.position.x, output.map_pose.position.y, output.map_pose.position.z = matrix[:3, 3]
                for key, value in zip(('x','y','z','w'), quaternion_from_matrix(matrix)):
                    setattr(output.map_pose.orientation, key, value)
            state_pub.publish(output)
            if path_pub.get_num_connections():
                identity = (output.session_id,output.generation)
                if identity != path_identity[0] or output.health != output.VALID:
                    local_path.poses.clear()
                path_identity[0] = identity
                local_path.header = output.header
                if output.health == output.VALID:
                    local_path.poses.append(PoseStamped(header=output.header,pose=output.pose))
                    if len(local_path.poses)>2000:
                        del local_path.poses[:-2000]
                path_pub.publish(local_path)
            binding = MapAssociation()
            binding.header = output.header
            binding.session_id, binding.generation = output.session_id, output.generation
            binding.map_id, binding.asset_fingerprint = map_id, fingerprint
            for key in ('valid', 'map_frame', 'association_session', 'association_generation', 'map_revision', 'status', 'reason'):
                setattr(binding, key, association[key])
            binding.coverage_status = association['coverage_status']
            binding.coverage_fingerprint = association['coverage_fingerprint'] or ''
            if mode == 'mapping':
                binding.map_id = 'mapping:'+binding.association_session if binding.association_session else ''
            if association['measurement_stamp_s'] is not None:
                binding.has_measurement_stamp = True
                binding.measurement_stamp = rospy.Time.from_sec(association['measurement_stamp_s'])
                binding.age_s = association['age_s']
            if association['valid']:
                binding.map_from_local = transform(association['map_from_local'], association['map_frame'],
                                                  output.header.frame_id, output.header.stamp).transform
            map_pub.publish(binding)
            if message.health == message.VALID:
                from perception.mapping_localization.context import pose_matrix
                local = pose_matrix(row['position'], row['quaternion_xyzw'])
                transforms.sendTransform(transform(local, output.header.frame_id, output.child_frame_id, output.header.stamp))
                if association['valid']:
                    transforms.sendTransform(transform(association['map_from_local'], association['map_frame'],
                                                       output.header.frame_id, output.header.stamp))

    subscribers = [rospy.Subscriber('/localization/frontend_state', LocalizationState, frontend, queue_size=16),
                   rospy.Subscriber('/localization/map_state', LocalizationState, match, queue_size=16)]
    rospy.spin()
