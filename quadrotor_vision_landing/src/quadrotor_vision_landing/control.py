import time
import numpy as np
import math
from scipy.spatial.transform import Rotation


class Control:

    def __init__(self):
        # Camera parameters
        self.horizontal_FOV = 69.4
        self.vertical_FOV = 42.5
        self.camera_A = np.array([[606.3280029296875, 0.0, 315.54412841796875],
                                  [0.0, 606.9010620117188, 242.91458129882812],
                                  [0.0, 0.0, 1.0]])
        self.fx = self.camera_A[0][0]
        self.fy = self.camera_A[1][1]
        self.cx = self.camera_A[0][2]
        self.cy = self.camera_A[1][2]

        # Camera translation
        camera_x_offset = -0.035
        camera_y_offset = 0.145
        camera_z_offset = -0.105
        cam_roll = 180
        camera_transform_rx = Rotation.from_euler('x', cam_roll, degrees=True).as_matrix()
        cam_pitch = 0
        camera_transform_ry = Rotation.from_euler('y', cam_pitch, degrees=True).as_matrix()
        cam_yaw = -90
        camera_transform_rz = Rotation.from_euler('z', cam_yaw, degrees=True).as_matrix()
        camera_transform_r = camera_transform_rx @ camera_transform_ry @ camera_transform_rz
        self.h_cam = np.array([[np.nan, np.nan, np.nan, camera_x_offset],
                               [np.nan, np.nan, np.nan, camera_y_offset],
                               [np.nan, np.nan, np.nan, camera_z_offset],
                               [0, 0, 0, 1]])
        self.h_cam[:3, :3] = camera_transform_r

        # Velocity capping parameters
        self.last_x = 0
        self.last_y = 0
        self.last_z = 0
        self.max_delta_v = 0.2
        self.max_x_vel = 0.2
        self.max_y_vel = 0.2
        self.max_z_vel = 0.2

        # Helipad measurements (in metres)
        self.helipad_height = 0.25
        self.helipad_width = 0.20

        # Image servoing
        self.vel_scalar = 0.0015
        self.vel_vector = np.array([
            [self.vel_scalar],
            [self.vel_scalar],
            [0.5 * self.vel_scalar],
            [self.vel_scalar],
            [self.vel_scalar],
            [self.vel_scalar],
        ])
        self.desired_p1 = np.array([[128], [0]]).astype(int)
        self.desired_p2 = np.array([[128], [480]]).astype(int)
        self.desired_p3 = np.array([[512], [0]]).astype(int)
        self.desired_p4 = np.array([[512], [480]]).astype(int)

        # Kalman filter parameters
        self.last_time = None
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.y = None
        self.y_rows_to_keep = 50  # Keep last 50 rows in y
        self.S = None
        self.K = None
        self.error = None
        self.x_hat = None
        p0 = 100
        self.P = p0 * np.eye(4)
        self.Q = 0.000001 * np.eye(4)
        self.kf_initialised = False
        self.kf_iteration_count = 0

    def kalman_filter(self, x, y):
        current_time = time.time()
        if self.x_hat is None:
            self.x_hat = np.array([[x], [y], [0], [0]])
            self.last_time = current_time
            self.y = np.array([[x, y]])
            return None, self.kf_initialised
        else:
            self.y = np.append(self.y, np.array([[x, y]]), axis=0)
            self.y = self.y[-self.y_rows_to_keep:, :]
            r = np.cov(np.transpose(self.y))
            dt = current_time - self.last_time
            a = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

            self.x_hat = a @ self.x_hat
            self.P = a @ self.P @ a.T + self.Q

            self.S = self.H @ self.P @ self.H.T + r
            self.K = self.P @ self.H.T @ np.linalg.inv(self.S)
            self.error = np.array([[x], [y]]) - (self.H @ self.x_hat)
            self.x_hat = self.x_hat + (self.K @ self.error)
            self.P = self.P - (self.K @ self.H @ self.P)
            
            self.last_time = current_time

            if not self.kf_initialised:
                self.kf_iteration_count += 1
                if self.kf_iteration_count > 10:
                    self.kf_initialised = True
            return self.x_hat, self.kf_initialised

    def create_jacobian(self, u, v, z):
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        return np.array([
            [-1 / z, 0, x / z, x * y, -(1 + x ** 2), y],
            [0, -1 / z, y / z, 1 + y ** 2, -x * y, -x]
        ])

    def pixels_to_3d(self, u, v, d):
        x_over_z = (u - self.cx) / self.fx
        y_over_z = (self.cy - v) / self.fy
        z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
        x = x_over_z * z
        y = y_over_z * z
        return y, x, z

    @staticmethod
    def get_rotation_angles(quad_pose):
        orient = quad_pose.pose.orientation
        r = Rotation.from_quat([orient.x, orient.y, orient.z, orient.w])
        r, p, y = r.as_euler('xyz', degrees=True)
        if (y > 90) & (y < 180):
            heading = 360 - (y - 90)
        else:
            heading = 90 - y

        if (heading > 0) & (heading < 180):
            heading = - heading
        else:
            heading = abs(360 - heading)

        return r, -1*p, heading

    def pixels_to_local(self, u, v, depth, quad_pose):
        rx, ry, rz = self.get_rotation_angles(quad_pose)
        x, y, z = self.pixels_to_3d(u, v, depth)
        h_measure = np.array([[x], [y], [z], [1]])
        quad_rx = Rotation.from_euler('x', rx, degrees=True).as_matrix()
        quad_ry = Rotation.from_euler('y', ry, degrees=True).as_matrix()
        quad_rz = Rotation.from_euler('z', rz, degrees=True).as_matrix()
        quad_rxyz = quad_rx @ quad_ry @ quad_rz
        h_quad = np.array([[np.nan, np.nan, np.nan, quad_pose.pose.position.x],
                           [np.nan, np.nan, np.nan, quad_pose.pose.position.y],
                           [np.nan, np.nan, np.nan, quad_pose.pose.position.z],
                           [0, 0, 0, 1]])
        h_quad[:3, :3] = quad_rxyz
        global_position = self.h_cam @ h_measure
        global_position = h_quad @ global_position
        return global_position[0][0], global_position[1][0], global_position[2][0], rx, ry, rz

    def global_to_body(self, x, y, z, quad_pose):
        rx, ry, rz = self.get_rotation_angles(quad_pose)
        quad_rx = Rotation.from_euler('x', rx, degrees=True).as_matrix()
        quad_ry = Rotation.from_euler('y', ry, degrees=True).as_matrix()
        quad_rz = Rotation.from_euler('z', rz, degrees=True).as_matrix()
        quad_rxyz = quad_rx @ quad_ry @ quad_rz
        h_quad = np.array([[np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [0, 0, 0, 1]])
        h_quad[:3, :3] = quad_rxyz
        h_global = np.array([[x],
                             [y],
                             [z],
                             [1]])
        body = np.linalg.inv(h_quad) @ h_global
        return body[0][0], body[1][0]

    def visual_servoing(self, bounding_box, depth, image_width):
        x, y, w, h = bounding_box

        p1 = np.array([[x], [y]]).astype(int)
        p1_error = np.array([[int(p1[0] - self.desired_p1[0])],
                             [int(p1[1] - self.desired_p1[1])]])

        p2 = np.array([[x], [y + h]]).astype(int)
        p2_error = np.array([[int(p2[0] - self.desired_p2[0])],
                             [int(p2[1] - self.desired_p2[1])]])

        p3 = np.array([[x + w], [y]]).astype(int)
        p3_error = np.array([[int(p3[0] - self.desired_p3[0])],
                             [int(p3[1] - self.desired_p3[1])]])

        p4 = np.array([[x + w], [y + h]]).astype(int)
        p4_error = np.array([[int(p4[0] - self.desired_p4[0])],
                             [int(p4[1] - self.desired_p4[1])]])

        try:
            p1_depth = depth[p1[0][0], p1[1][0]][0] / 1000
            p2_depth = depth[p2[0][0], p2[1][0]][0] / 1000
            p3_depth = depth[p3[0][0], p3[1][0]][0] / 1000
            p4_depth = depth[p4[0][0], p4[1][0]][0] / 1000
            centre_depth = depth[x+w/2, y+h/2] / 1000
            if (p1_depth == 0) | (p2_depth == 0) | (p3_depth == 0) | (p4_depth == 0):
                image_height = depth.shape[0]
                image_height_m = image_height * self.helipad_height / h
                depth = (0.5 * image_height_m) / math.tan(math.radians(0.5 * self.vertical_FOV))
                p1_depth = depth
                p2_depth = depth
                p3_depth = depth
                p4_depth = depth
                centre_depth = depth
        except (IndexError, TypeError):
            image_width_m = image_width * self.helipad_width / w
            depth = (0.5 * image_width_m) / math.tan(math.radians(0.5 * self.horizontal_FOV))
            p1_depth = depth
            p2_depth = depth
            p3_depth = depth
            p4_depth = depth
            centre_depth = depth

        p1_jacobian = self.create_jacobian(p1[0][0], p1[1][0], p1_depth)
        p1_desired_jacobian = self.create_jacobian(self.desired_p1[0][0], self.desired_p1[1][0], p1_depth)

        p2_jacobian = self.create_jacobian(p2[0][0], p2[1][0], p2_depth)
        p2_desired_jacobian = self.create_jacobian(self.desired_p2[0][0], self.desired_p2[1][0], p2_depth)

        p3_jacobian = self.create_jacobian(p3[0][0], p3[1][0], p3_depth)
        p3_desired_jacobian = self.create_jacobian(self.desired_p3[0][0], self.desired_p3[1][0], p3_depth)

        p4_jacobian = self.create_jacobian(p4[0][0], p4[1][0], p4_depth)
        p4_desired_jacobian = self.create_jacobian(self.desired_p4[0][0], self.desired_p4[1][0], p2_depth)

        error = np.concatenate((p1_error, p2_error, p3_error, p4_error))
        point_jacobians = np.concatenate((p1_jacobian, p2_jacobian, p3_jacobian, p4_jacobian))
        desired_point_jacobians = np.concatenate((p1_desired_jacobian, p2_desired_jacobian, p3_desired_jacobian,
                                                  p4_desired_jacobian))

        le = 0.5 * np.linalg.pinv(point_jacobians + desired_point_jacobians)
        output_vels = -self.vel_vector * np.matmul(le, error)
        points = [{'point': p1, 'desired_point': self.desired_p1, 'depth': p1_depth, 'number': 1},
                  {'point': p2, 'desired_point': self.desired_p2, 'depth': p2_depth, 'number': 2},
                  {'point': p3, 'desired_point': self.desired_p3, 'depth': p3_depth, 'number': 3},
                  {'point': p4, 'desired_point': self.desired_p4, 'depth': p4_depth, 'number': 4}]
        output_vels = np.around(output_vels, decimals=3)

        x = output_vels[0][0]
        y = -output_vels[1][0]
        z = -output_vels[2][0]
        if abs(x - self.last_x) > self.max_delta_v:
            if x > 0:
                x = self.last_x + self.max_delta_v
            else:
                x = self.last_x - self.max_delta_v

        if abs(y - self.last_y) > self.max_delta_v:
            if y > 0:
                y = self.last_y + self.max_delta_v
            else:
                y = self.last_y - self.max_delta_v

        if abs(z - self.last_z) > self.max_delta_v:
            if z > 0:
                z = self.last_z + self.max_delta_v
            else:
                z = self.last_z - self.max_delta_v

        self.last_x = x
        self.last_y = y
        self.last_z = z
        return x, y, z, 0, 0, 0, points, centre_depth

    @staticmethod
    def pid(kp, ki, kd):
        return PID(kp, ki, kd)


class PID:

    def __init__(self, kp, ki, kd):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.setpoint = None
        self.last_time = None
        self.last_error = None
        self.i_term_sum = 0

    def set_setpoint(self, setpoint):
        self.setpoint = setpoint
        return self

    def generate_output(self, feedback):

        current_time = time.time()

        if self.setpoint is None:
            raise ValueError('No setpoint value set')

        error = self.setpoint - feedback

        p_term = error * self.Kp
        d_term = 0
        if self.last_time is not None:
            time_delta = current_time - self.last_time
            self.i_term_sum += error * time_delta

            if self.last_error is not None:
                error_delta = error - self.last_error
                d_term = error_delta / time_delta

        # Integral windup protection, if error = 0 or error crosses zero, reset integral sum
        if self.last_error is not None:
            if (error == 0) | ((error > 0) & (self.last_error < 0)) | \
                    ((error < 0) & (self.last_error > 0)):
                self.i_term_sum = 0

        output_value = (p_term +
                        (self.Ki * self.i_term_sum) +
                        (self.Kd * d_term))

        self.last_time = current_time
        self.last_error = error

        return output_value
