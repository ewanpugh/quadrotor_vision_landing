import time
import numpy as np


class Control:

    def __init__(self):
        self.vel_scalar = 0.015
        self.focal_length = 0.128
        self.pixel_ratio = 1
        self.cu = 128
        self.cv = 72

    @staticmethod
    def pid(kp, ki, kd):
        return PID(kp, ki, kd)

    def calculate_x(self, u):
        u = int(u)
        return (u - self.cu) / (self.focal_length * self.pixel_ratio)

    def calculate_y(self, v):
        v = int(v)
        return (v - self.cv) / self.focal_length

    def create_jacobian(self, u, v, z):
        x = self.calculate_x(u)
        y = self.calculate_y(v)
        return np.array([
            [-1 / z, 0, x / z, x * y, -(1 + x ** 2), y],
            [0, -1 / z, y / z, 1 + y ** 2, -x * y, -x]
        ])

    def visual_servoing(self, bounding_box, depth_image):
        x, y, w, h = bounding_box

        p1 = np.array([[x], [y]]).astype(int)
        desired_p1 = np.array([[self.cu-(w/2)], [self.cv-(h/2)]]).astype(int)
        # desired_p1 = np.array([[71], [0]]).astype(int)
        p1_error = np.array([[int(p1[0] - desired_p1[0])],
                             [int(p1[1] - desired_p1[1])]])
        p1_depth = depth_image[p1[1][0], p1[0][0]]
        p1_jacobian = self.create_jacobian(p1[0][0], p1[1][0], p1_depth)
        p1_desired_jacobian = self.create_jacobian(desired_p1[0][0], desired_p1[1][0], p1_depth)

        p2 = np.array([[x], [y+h]]).astype(int)
        desired_p2 = np.array([[self.cu-(w/2)], [self.cv+(h/2)]]).astype(int)
        # desired_p2 = np.array([[71], [144]]).astype(int)
        p2_error = np.array([[int(p2[0] - desired_p2[0])],
                             [int(p2[1] - desired_p2[1])]])
        p2_depth = depth_image[p2[1][0], p2[0][0]]
        p2_jacobian = self.create_jacobian(p2[0][0], p2[1][0], p2_depth)
        p2_desired_jacobian = self.create_jacobian(desired_p2[0][0], desired_p2[1][0], p2_depth)

        p3 = np.array([[x+w], [y]]).astype(int)
        desired_p3 = np.array([[self.cu+(w/2)], [self.cv-(h/2)]]).astype(int)
        # desired_p3 = np.array([[185], [0]]).astype(int)
        p3_error = np.array([[int(p3[0] - desired_p3[0])],
                             [int(p3[1] - desired_p3[1])]])
        p3_depth = depth_image[p3[1][0], p3[0][0]]
        p3_jacobian = self.create_jacobian(p3[0][0], p3[1][0], p3_depth)
        p3_desired_jacobian = self.create_jacobian(desired_p3[0][0], desired_p3[1][0], p3_depth)

        p4 = np.array([[x + w], [y+h]]).astype(int)
        desired_p4 = np.array([[self.cu + (w/2)], [self.cv + (h/2)]]).astype(int)
        # desired_p4 = np.array([[185], [144]]).astype(int)
        p4_error = np.array([[int(p4[0] - desired_p4[0])],
                             [int(p4[1] - desired_p4[1])]])
        p4_depth = depth_image[p4[1][0], p4[0][0]]
        p4_jacobian = self.create_jacobian(p4[0][0], p4[1][0], p4_depth)
        p4_desired_jacobian = self.create_jacobian(desired_p4[0][0], desired_p4[1][0], p2_depth)

        error = np.concatenate((p1_error, p2_error, p3_error, p4_error))
        point_jacobians = np.concatenate((p1_jacobian, p2_jacobian, p3_jacobian, p4_jacobian))
        desired_point_jacobians = np.concatenate((p1_desired_jacobian, p2_desired_jacobian, p3_desired_jacobian,
                                                  p4_desired_jacobian))
        le = 0.5*np.linalg.pinv(point_jacobians+desired_point_jacobians)
        output_vels = -self.vel_scalar * np.matmul(le, error)
        points = {'p1': p1, 'desired_p1': desired_p1,
                  'p2': p2, 'desired_p2': desired_p2,
                  'p3': p3, 'desired_p3': desired_p3,
                  'p4': p4, 'desired_p4': desired_p4}

        return output_vels, points


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
