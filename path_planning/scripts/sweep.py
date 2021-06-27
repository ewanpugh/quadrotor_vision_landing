#!/home/njnano1/python36_ws/py36env/bin/python3.6

import rospy
from quadrotor_vision_landing.px4 import PX4
from quadrotor_vision_landing.computer_vision import ComputerVision
from quadrotor_vision_landing.control import Control
from mavros_msgs.msg import Waypoint
import math
import matplotlib.pyplot as plt
import time
import numpy as np


#########################################
############### FUNCTIONS ###############
#########################################


def cap_velocity(vel, max_velocity):
    if (vel > 0) & (abs(vel) > max_velocity):
        vel = max_velocity
    elif (vel < 0) & (abs(vel) > max_velocity):
        vel = -max_velocity
    return vel


def euler_to_quaternion(roll, pitch, yaw):
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
        yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
        yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)

    return [qx, qy, qz, qw]


def quaternion_to_euler(x, y, z, w):

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]


#########################################
################# SCRIPT ################
#########################################


helipad_detector = ComputerVision()
control = Control()
px4 = PX4()
quad_state = px4.quadcopter_state()
quad_control = px4.quadcopter_control()
procedures = px4.procedures()

horizontal_FOV = 64
vertical_FOV = 41

coord_1_string = input('Enter top left co-ordinates for search sweep in format "x, y": ')
if ' ' in coord_1_string:
    coord_1 = coord_1_string.split(', ')
else:
    coord_1 = coord_1_string.split(',')
coord_1 = [float(x) for x in coord_1]

coord_2_string = input('Enter bottom right co-ordinates for search sweep in format "x, y": ')
if ' ' in coord_2_string:
    coord_2 = coord_2_string.split(', ')
else:
    coord_2 = coord_2_string.split(',')
coord_2 = [float(x) for x in coord_2]

altitude = float(input('What altitude for search sweep (in metres)?: '))

visible_width = 2 * altitude * math.tan(math.radians(horizontal_FOV / 2))
width_with_overlap = visible_width * 0.8

start_y = coord_1[1] - width_with_overlap / 2
points = [{'x': coord_1[0], 'y': start_y, 'z': altitude},
          {'x': coord_2[0], 'y': start_y, 'z': altitude}]
current_point = 2
at_bottom = 0
while at_bottom < 2:
    previous_x = points[-1]['x']
    previous_y = points[-1]['y']
    if current_point == 1:
        new_x = coord_2[0]
        new_y = previous_y
        current_point += 1
    elif current_point == 2:
        new_x = previous_x
        new_y = previous_y - width_with_overlap
        current_point += 1
    elif current_point == 3:
        new_x = coord_1[0]
        new_y = previous_y
        current_point += 1
    else:
        new_x = previous_x
        new_y = previous_y - width_with_overlap
        current_point = 1
    if (new_y - width_with_overlap / 2) < coord_2[1]:
        at_bottom += 1
        if new_y < coord_2[1]:
            new_y = coord_2[1]

    points.append({'x': new_x, 'y': new_y, 'z': altitude})
points.append({'x': 0, 'y': 0, 'z': altitude})

print("Points: ")
for point in points:
    print("x: {}, y: {}, z: {}".format(point['x'], point['y'], point['z']))
_ = input("Press enter to confirm points ")

current_pose = quad_state.current_pose().pose.position
current_x = current_pose.x
current_y = current_pose.y
current_z = current_pose.z
rospy.loginfo('Waiting for offboard mode')
while quad_state.current_state().mode.lower() != 'offboard':
    quad_control.publish_pose(x = current_x, y = current_y, z = current_z)

rotation_rate = 15  # deg/s
max_vel = 0.2
accept_pose_dif = 0.1

for i in range(0, len(points)):
    desired_pose = points[i]
    rospy.loginfo('Flying to point {}'.format(i + 1))
    rospy.loginfo('x: {}, y: {}'.format(desired_pose['x'], desired_pose['y']))
    not_at_desired = True

    pose = quad_state.current_pose().pose.position
    x = desired_pose['x'] - pose.x
    y = desired_pose['y'] - pose.y
    z = desired_pose['z'] - pose.z
    while not_at_desired:
        if (abs(x) < accept_pose_dif) & (abs(y) < accept_pose_dif) & (abs(z) < accept_pose_dif):
            not_at_desired = False
        pose = quad_state.current_pose().pose.position
        x = desired_pose['x'] - pose.x
        y = desired_pose['y'] - pose.y
        z = desired_pose['z'] - pose.z
        quad_control.publish_pose(x=desired_pose['x'], y=desired_pose['y'], z=desired_pose['z'])

    start_time = time.time()
    while time.time() - start_time < 1:
        quad_control.publish_pose(x=desired_pose['x'], y=desired_pose['y'], z=desired_pose['z'])
