#!/home/njnano1/python36_ws/py36env/bin/python3.6

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from quadrotor_vision_landing.control import Control
from quadrotor_vision_landing.computer_vision import ComputerVision
import time
import math
import numpy as np
import cv2
import copy
from datetime import datetime
from statistics import mean

img_raw = None
depth_raw = None
current_pose = None


def rotate(x1, y1, ang):
    x2 = x1*math.cos(ang) + y1*math.sin(ang)
    y2 = -x1*math.sin(ang) + y1*math.cos(ang)
    return x2, y2


def yaw_from_quaternion(orientation):
    qx = orientation.x
    qy = orientation.y
    qz = orientation.z
    qw = orientation.w
    t3 = +2.0 * (qw * qz + qx * qy)
    t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
    yaw_z = math.atan2(t3, t4)
    return yaw_z - math.radians(90)  # in radians


def draw_points_on_image(pnts, im):
    if pnts is not None:
        for point in pnts:
            if point['number'] == 1:
                color = (255, 0, 0)
            elif point['number'] == 2:
                color = (0, 255, 0)
            elif point['number'] == 3:
                color = (0, 0, 255)
            else:
                color = (100, 100, 100)
            im = cv2.circle(im, (int(point['point'][0][0]), int(point['point'][1][0])), radius=3, color=color, thickness=-1)
            im = cv2.circle(im, (int(point['desired_point'][0][0]), int(point['desired_point'][1][0])), radius=3, color=color, thickness = -1)
    return im


def rgb_cb(image_data):
    global img_raw
    img_raw = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)


def depth_cb(image_data):
    global depth_raw
    depth_raw = np.frombuffer(image_data.data, dtype=np.uint16).reshape(image_data.height, image_data.width, -1)


def pose_cb(msg):
    global current_pose
    current_pose = msg


def write_to_csv(message, method):
    global csv_fname
    with open('/home/njnano1/Desktop/kalman_testing/{}'.format(csv_fname), method) as f:
        f.write("{}\n".format(message))


rospy.init_node('node', anonymous=True)
rate = rospy.Rate(20)

rgb_sub = rospy.Subscriber('/camera/color/image_raw', Image, rgb_cb)
depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, depth_cb)
pose_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, pose_cb)
control = Control()
helipad_detector = ComputerVision()

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
x = None
y = None
z = None
x_ang = None
y_ang = None
z_ang = None
points = None
landed = False

time.sleep(1)  # Give subscribers time to register
now = datetime.now()
csv_fname = now.strftime("%Y-%m-%d-%H-%M-%S.csv")
write_to_csv("kf_x,kf_y,x_raw,y_raw,z_raw,kf_x_dot,kf_y_dot,rx,ry,rz,helipad_x_dot_body,helipad_y_dot_body", "w")

camera_x_offset = -0.045
camera_y_offset = 0.08




while True:
    try:
        quad_pose = copy.copy(current_pose)
        img = copy.copy(img_raw)
        depth = copy.copy(depth_raw)
        if (img is None) | (depth is None):
            raise IndexError('No image captured')
        if not final_landing_stage:
            if not tracker_initiated:  # Detecting helipad for the first time
                detected_boolean, helipad_centroid, bb = helipad_detector.geometry_helipad_detection(img)
                if detected_boolean:  # If detected
                    initiate_tracker = True  # Tell script to initiate tracker
                    if initiate_tracker & (not tracker_initiated):  # Initiate tracker
                        helipad_detector.init_tracker(img, bb)
                        tracker_initiated = True
            else:  # Runs once tracker initiated
                if tracker_frames % cross_val_frames == 0:  # Every n frames, re-initialise tracker
                    detected_boolean, helipad_centroid, bb = helipad_detector.geometry_helipad_detection(img)
                    if detected_boolean:
                        helipad_detector.init_tracker(img, bb)
                        tracker_frames += 1
                    else:
                        bb = helipad_detector.track_object(img)
                else:  # Just use tracker
                    bb = helipad_detector.track_object(img)
                    tracker_frames += 1

            if bb:
                x, y, w, h = bb
                if h / img.shape[0] > 0.75:
                    final_landing_stage = True

                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                if (bb[0] != 0) & (bb[1] != 0):
                    x_vel, y_vel, z_vel, _, _, _, points, centre_depth = control.visual_servoing(bb, depth, img.shape[1])
                    x_global, y_global, z_global, rx, ry, rz = control.pixels_to_local(x+w/2, y+h/2, quad_pose.pose.position.z, quad_pose)  # Pixels to 3D distance
                    estimate, kf_initialised = control.kalman_filter(x_global, y_global)
                    if estimate is not None:
                        x_est = estimate[0][0]
                        y_est = estimate[1][0]
                        x_dot_est = estimate[2][0]
                        y_dot_est = estimate[3][0]
                        x_body, y_body = control.global_to_body(x_dot_est, y_dot_est, 0, quad_pose)
                        write_to_csv("{},{},{},{},{},{},{},{},{},{},{},{}".format(x_est, y_est, x_global, y_global, z_global,x_dot_est,y_dot_est,rx,ry,rz,x_body, y_body), "a")
                    if (x_vel is not None) & (y_vel is not None) & (z_vel is not None):
                        # print(
                        #     "x: {:.2f}, y: {:.2f}, z: {:.2f}, x_ang: {:.2f}, y_ang: {:.2f}, z_ang: {:.2f}".format(
                        #         x_vel, y_vel, z_vel, x_ang_vel, y_ang_vel, z_ang_vel))
                        img = draw_points_on_image(points, img)

        cv2.imshow('Image', img)
        cv2.waitKey(1)
    except IndexError as e:
        print(e)

