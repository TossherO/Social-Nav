cmake_minimum_required(VERSION 3.0.2)
project(wheelchair_sensors)
find_package(catkin REQUIRED COMPONENTS rospy sensor_msgs nav_msgs geometry_msgs std_msgs)
catkin_python_setup()
catkin_package()
catkin_install_python(PROGRAMS scripts/chassis_feedback_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY config launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
