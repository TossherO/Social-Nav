#!/usr/bin/env python3
"""Explicitly enabled serial receive-only feedback; never sends control frames."""
import math
import time
import rospy
import serial
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, Int8, UInt16
from wheelchair_sensors.feedback import FrameDecoder

def main():
    rospy.init_node('chassis_feedback')
    if not rospy.get_param('~allow_serial_read', False):
        rospy.logfatal('Serial access disabled. Enable allow_serial_read only for a stationary feedback test.')
        return 1
    device = rospy.get_param('~port', '/dev/wheelchair_chassis')
    baudrate = rospy.get_param('~baudrate', 19200)
    timeout = rospy.get_param('~feedback_timeout', 2.0)
    legacy_z = rospy.get_param('~legacy_theta_in_z', True)
    odom_pub = rospy.Publisher('/odom_encoder', Odometry, queue_size=10)
    battery_pub = rospy.Publisher('/robot_battery', Float32, queue_size=10)
    battery_raw_pub = rospy.Publisher('/robot_battery_raw', UInt16, queue_size=10)
    battery_valid_pub = rospy.Publisher('/robot_battery_valid', Bool, queue_size=10)
    state_pub = rospy.Publisher('/robot_state_back', Int8, queue_size=10)
    joystick_pub = rospy.Publisher('/joystick', Twist, queue_size=10)
    velocity_pub = rospy.Publisher('/robot_vel', Twist, queue_size=10)
    communication_pub = rospy.Publisher('/communacation', Bool, queue_size=10)
    battery_ok_pub = rospy.Publisher('/low_battery', Bool, queue_size=10)
    port = serial.Serial(port=None, baudrate=baudrate, timeout=0.2, exclusive=True)
    port.dtr = False
    port.rts = False
    port.port = device
    decoder = FrameDecoder()
    last_received = None
    started = time.monotonic()
    next_status = started
    try:
        port.open()
        rospy.loginfo('Receive-only feedback on %s at %d baud; no serial writes or /cmd_vel subscription', device, baudrate)
        while not rospy.is_shutdown():
            chunk = port.read(max(1, min(port.in_waiting, 4096)))
            for feedback in decoder.feed(chunk):
                last_received = time.monotonic()
                odom = Odometry()
                odom.header.stamp = rospy.Time.now()
                odom.header.frame_id = rospy.get_param('~frame_id', 'odom')
                odom.child_frame_id = rospy.get_param('~child_frame_id', 'base_footprint')
                odom.pose.pose.position.x = feedback.x
                odom.pose.pose.position.y = feedback.y
                # Retain the old interface by default; use false for standard planar z=0.
                odom.pose.pose.position.z = feedback.theta if legacy_z else 0.0
                odom.pose.pose.orientation.z = math.sin(feedback.theta / 2.0)
                odom.pose.pose.orientation.w = math.cos(feedback.theta / 2.0)
                odom.twist.twist.linear.x = feedback.linear_velocity
                odom.twist.twist.angular.z = feedback.angular_velocity
                odom_pub.publish(odom)
                battery_raw_pub.publish(feedback.battery)
                battery_valid = 0 <= feedback.battery <= 100
                battery_valid_pub.publish(battery_valid)
                if battery_valid:
                    battery_pub.publish(feedback.battery)
                    battery_ok_pub.publish(feedback.battery >= 20)
                else:
                    rospy.logwarn_throttle(5.0, 'Battery field %d is outside 0..100; publishing only raw/validity diagnostics', feedback.battery)
                state_pub.publish(feedback.state)
                joy = Twist()
                joy.linear.x, joy.angular.z = feedback.joystick_linear, feedback.joystick_angular
                joystick_pub.publish(joy)
                velocity_pub.publish(odom.twist.twist)
            now = time.monotonic()
            if now >= next_status:
                communication_pub.publish(last_received is not None and now - last_received < timeout)
                next_status = now + 1.0
                if last_received is None and now - started >= timeout:
                    rospy.logwarn_throttle(5.0, 'No autonomous feedback. This receiver cannot poll or control the chassis; protocol confirmation is required.')
    except (serial.SerialException, OSError) as error:
        rospy.logerr('Serial feedback failed: %s', error)
        return 1
    finally:
        if port.is_open:
            port.close()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
