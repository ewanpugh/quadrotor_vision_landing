cmake_minimum_required(VERSION 3.0.2)
project(quadrotor_vision_landing)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  mavros_msgs
)
catkin_python_setup()
catkin_package()
