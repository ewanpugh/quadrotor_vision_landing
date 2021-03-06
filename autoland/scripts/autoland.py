#!/home/njnano1/python36_ws/py36env/bin/python3.6


#################################################
#################### IMPORTS ####################
#################################################

import rospy
from quadrotor_vision_landing.px4 import PX4
from quadrotor_vision_landing.computer_vision import ComputerVision
from quadrotor_vision_landing.control import Control

import numpy as np
import math
import cv2
import time
import pandas as pd
import os
import sys
from datetime import datetime


#################################################
################### FUNCTIONS ###################
#################################################
now = datetime.now()
csv_fname = now.strftime("%Y-%m-%d-%H-%M-%S.csv")


def write_to_csv(message, method):
    global csv_fname
    with open('/home/njnano1/Desktop/kalman_testing/{}'.format(csv_fname), method) as f:
        f.write("{}\n".format(message))


def draw_points_on_image(pnts, im):
    if pnts is not None:
        for pnt in points:
            if pnt['number'] == 1:
                color = (255, 0, 0)
            elif pnt['number'] == 2:
                color = (0, 255, 0)
            elif pnt['number'] == 3:
                color = (0, 0, 255)
            else:
                color = (100, 100, 100)
            im = cv2.circle(im, (int(pnt['point'][0][0]), int(pnt['point'][1][0])), radius=3, color=color, thickness=-1)
            im = cv2.circle(im, (int(pnt['desired_point'][0][0]), int(pnt['desired_point'][1][0])), radius=3, color=color, thickness = -1)
    return im


def initialise_log_file(path):
    with open(log_path, 'w') as f:
        f.write("###################################################\n")
        f.write("##               autoland log file               ##\n")
        f.write("##    Autogenerated on {}    ##\n".format(datetime.now().strftime("%d-%m-%Y at %H:%M:%S")))
        f.write("###################################################\n\n")
        python_info = sys.version_info
        f.write("Python version: {}.{}.{}\n".format(python_info[0], python_info[1], python_info[2]))
        f.write("ROS Version: {}\n".format(os.popen('rosversion -d').read().strip('\n')))
        f.write("OpenCV Version: {}\n\n".format(cv2.__version__))


def log(path, message):
    timestamp = int(time.time())
    with open(path, 'a') as f:
        f.write("[{}] {}\n".format(timestamp, message))

#################################################
###################### MAIN #####################
#################################################


if __name__ == '__main__':

    parent_dir = os.popen('rospack find autoland').read().strip('\n')
    log_dir = parent_dir + '/logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file_name = 'autoland_{}.txt'.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    log_path = '{}/{}'.format(log_dir, log_file_name)
    initialise_log_file(log_path)

    rospy.loginfo('Initialising')
    log(log_path, "Initialising...")
    computer_vision = ComputerVision()
    log(log_path, "Initialised computer vision module!")

    control = Control()
    log(log_path, "Initialised control module!")

    px4 = PX4()
    log(log_path, "Initialised PX4 module!")

    quad_state = px4.quadcopter_state()
    log(log_path, "Initialised PX4 state module!")

    quad_control = px4.quadcopter_control()
    log(log_path, "Initialised PX4 control module!")

    image = computer_vision.get_rgb_image()
    if image is not None:
        log(log_path, "RealSense camera stream running!")
    image_height = image.shape[0]
    image_width = image.shape[1]
    log(log_path, "RealSense camera resolution: {}x{}".format(image_width, image_height))

    detected_boolean = False
    initiate_tracker = False
    tracker_initiated = False
    cross_val_frames = 10  # Cross validate tracker with detection every 10 frames
    tracker_frames = 0
    final_landing_stage = False
    bb = None

    img = None
    depth = None
    helipad_centroid = None
    bb = None
    x_vel = None
    y_vel = None
    z_vel = None
    x_ang_vel = None
    y_ang_vel = None
    z_ang_vel = None
    points = None
    p1_depth = None
    landed = False
    log(log_path, "Initialisation finished!")

    horizontal_FOV = control.horizontal_FOV
    vertical_FOV = control.vertical_FOV

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

    quad_control.set_param('MPC_XY_VEL_MAX', 0.2)

    write_to_csv("kf_x,kf_y,hel_x,hel_y,hel_z,kf_x_dot,kf_y_dot,hel_x_body,hel_y_body", "w")

    current_pose = quad_state.current_pose().pose.position
    current_x = current_pose.x
    current_y = current_pose.y
    current_z = current_pose.z
    rospy.loginfo('Waiting for offboard mode')
    while quad_state.current_state().mode.lower() != 'offboard':
        quad_control.publish_pose(x=current_x, y=current_y, z=current_z)

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
            img = computer_vision.get_rgb_image()
            if img is not None:
                detected_boolean, helipad_centroid, bb = computer_vision.geometry_helipad_detection(img)
            if detected_boolean:
                break
        if detected_boolean:
            break

        start_time = time.time()

        while time.time() - start_time < 1:
            quad_control.publish_pose(x=desired_pose['x'], y=desired_pose['y'], z=desired_pose['z'])
            img = computer_vision.get_rgb_image()
            if img is not None:
                detected_boolean, helipad_centroid, bb = computer_vision.geometry_helipad_detection(img)
            if detected_boolean:
                break
        if detected_boolean:
            break

    if detected_boolean:
        computer_vision.init_tracker(img, bb)
        helipad_found = True
    else:
        helipad_found = False

    rospy.loginfo('Running')
    while helipad_found:
        try:
            img = computer_vision.get_rgb_image()
            depth = computer_vision.get_depth_image()
            quad_pose = quad_state.current_pose()
            if (img is None) | (depth is None):
                raise TypeError('No image captured')
            if not final_landing_stage:
                if tracker_frames % cross_val_frames == 0:  # Every n frames, re-initialise tracker
                    detected_boolean, helipad_centroid, bb = computer_vision.geometry_helipad_detection(img)
                    if detected_boolean:
                        computer_vision.init_tracker(img, bb)
                        tracker_frames += 1
                    bb = computer_vision.track_object(img)
                else:  # Just use tracker
                    bb = computer_vision.track_object(img)
                    tracker_frames += 1
                if bb:
                    x, y, w, h = bb
                    if quad_pose.pose.position.z < 0.75:
                        final_landing_stage = True

                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    if (bb[0] != 0) & (bb[1] != 0):
                        computer_vision.publish_bounding_box(bb)
                        x_vel, y_vel, z_vel, x_ang_vel, y_ang_vel, z_ang_vel, points, centre_depth = control.visual_servoing(bb, depth, image_width)
                        hel_x, hel_y, hel_z, rx, ry, rz = control.pixels_to_local(x + w / 2, y + h / 2,
                                                                                  quad_pose.pose.position.z - 0.11,
                                                                                  quad_pose)  # Pixels to 3D distance
                        estimate, kf_initialised = control.kalman_filter(hel_x, hel_y)
                        if estimate is not None:
                            x_est = estimate[0][0]
                            y_est = estimate[1][0]
                            x_dot_est = estimate[2][0]
                            y_dot_est = estimate[3][0]

                        if (x_vel is not None) & (y_vel is not None) & (z_vel is not None):
                            if kf_initialised & (estimate is not None):
                                x_vel_body, y_vel_body = control.global_to_body(x_dot_est, y_dot_est, 0, quad_pose)
                                x_vel += x_vel_body
                                y_vel += y_vel_body
                                write_to_csv(
                                    "{},{},{},{},{},{},{},{},{}".format(x_est, y_est, hel_x, hel_y, hel_z, x_dot_est,
                                                                        y_dot_est, x_vel_body, y_vel_body), "a")
                            print("x: {:.2f}, y: {:.2f}".format(x_vel, y_vel))
                            quad_control.publish_velocity(x_vel, y_vel, -0.1, 0, 0, 0)
                            img = draw_points_on_image(points, img)
                        else:
                            quad_control.publish_velocity()
                    else:
                        quad_control.publish_velocity()
                else:
                    quad_control.publish_velocity()
            else:
                z = -0.3
                quad_control.publish_velocity(x=x_vel_body, y=y_vel_body, z=z)

            if str(quad_state.current_state().armed).lower() == 'false':
                landed = True
                break
            # cv2.imshow('Image', img)
            # cv2.waitKey(1)
        except TypeError as e:
            print(e)
            quad_control.publish_velocity()

if not helipad_found:
    rospy.loginfo('Helipad not found')

if landed:
    rospy.loginfo('Landed!')
    log(log_path, "Landed!")
else:
    rospy.loginfo('Exiting...')
    log(log_path, "Exited")

cv2.destroyAllWindows()
rospy.loginfo('Script finished')
log(log_path, "Script finished!")
