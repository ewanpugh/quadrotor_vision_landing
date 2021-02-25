import rospy
import time
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from mavros_msgs.msg import State, ParamValue
from mavros_msgs.srv import CommandBool, SetMode, ParamGet, ParamSet

current_state = None
current_pose = None
current_velocity = None
start_pose = None


def state_cb(msg):
    global current_state
    current_state = msg


def pose_cb(msg):
    global current_pose
    global start_pose
    current_pose = msg
    if start_pose is None:
        start_pose = msg


def vel_cb(msg):
    global current_velocity
    current_velocity = msg


class PX4:
    def __init__(self, node_name='offb_node', rate=20):
        rospy.init_node(node_name, anonymous=True)
        self.rate = rospy.Rate(rate)

    @staticmethod
    def quadcopter_state():
        return QuadcopterState()

    @staticmethod
    def quadcopter_control():
        return QuadcopterCommand()

    @staticmethod
    def pid():
        return PID()

    @staticmethod
    def procedures():
        return Procedures()


class QuadcopterState(PX4):

    def __init__(self):
        super().__init__()
        self.state_sub = rospy.Subscriber('mavros/state', State, state_cb)
        self.pose_publisher = rospy.Subscriber('mavros/local_position/pose', PoseStamped, pose_cb)
        self.local_vel_sub = rospy.Subscriber('/mavros/local_position/velocity_body', TwistStamped, vel_cb)

    @staticmethod
    def current_state():
        global current_state
        while current_state is None:
            pass
        return current_state

    @staticmethod
    def current_pose():
        global current_pose
        while current_pose is None:
            pass
        return current_pose

    @staticmethod
    def start_pose():
        global start_pose
        while start_pose is None:
            pass
        return start_pose

    @staticmethod
    def current_velocity():
        global current_velocity
        while current_velocity is None:
            pass
        return current_velocity


class QuadcopterCommand(PX4):
    def __init__(self):
        super().__init__()
        self.pose_publisher = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.velocity_publisher = rospy.Publisher('mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=10)
        rospy.wait_for_service('mavros/cmd/arming')
        self.arm_service = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)
        rospy.wait_for_service('mavros/set_mode')
        self.mode_service = rospy.ServiceProxy('mavros/set_mode', SetMode)

    def publish_pose(self, x=None, y=None, z=None, x_orient=None, y_orient=None, z_orient=None, w_orient=None):
        pose_dict = {'x': x, 'y': y, 'z': z, 'x_orient': x_orient,
                     'y_orient': y_orient, 'z_orient': z_orient, 'w_orient': w_orient}
        for key in pose_dict.keys():
            if pose_dict[key] is None:
                pose_dict[key] = 0

        msg = PoseStamped()
        msg.pose.position.x = pose_dict['x']
        msg.pose.position.y = pose_dict['y']
        msg.pose.position.z = pose_dict['z']
        msg.pose.orientation.x = pose_dict['x_orient']
        msg.pose.orientation.y = pose_dict['y_orient']
        msg.pose.orientation.z = pose_dict['z_orient']
        msg.pose.orientation.w = pose_dict['w_orient']

        self.pose_publisher.publish(msg)

    def publish_velocity(self, x=None, y=None, z=None, x_ang=None, y_ang=None, z_ang=None):
        vel_dict = {'x': x, 'y': y, 'z': z, 'x_ang': x_ang, 'y_ang': y_ang, 'z_ang': z_ang}
        for key in vel_dict.keys():
            if vel_dict[key] is None:
                vel_dict[key] = 0

        msg = Twist()
        msg.linear.x = vel_dict['x']
        msg.linear.y = vel_dict['y']
        msg.linear.z = vel_dict['z']
        msg.angular.x = vel_dict['x_ang']
        msg.angular.y = vel_dict['y_ang']
        msg.angular.z = vel_dict['z_ang']
        self.velocity_publisher.publish(msg)

    def set_mode(self, mode):
        return self.mode_service(custom_mode=mode)

    def arm(self, command):
        return self.arm_service(command)


class PID(PX4):
    def __init__(self):
        super().__init__()
        rospy.wait_for_service('mavros/param/get')
        self.get_param_service = rospy.ServiceProxy('mavros/param/get', ParamGet)
        rospy.wait_for_service('mavros/param/set')
        self.set_param_service = rospy.ServiceProxy('mavros/param/set', ParamSet)

    def get_gains(self, axis):
        p_param = 'MC_{}RATE_P'.format(axis.upper())
        i_param = 'MC_{}RATE_I'.format(axis.upper())
        d_param = 'MC_{}RATE_D'.format(axis.upper())
        gains = {}
        for param in [p_param, i_param, d_param]:
            response = self.get_param_service(param_id=param)
            success = response.success
            value = response.value.real
            if success:
                gains[param] = value
            else:
                gains[param] = 'ERROR'
        return gains

    def set_gains(self, axis, p=None, i=None, d=None):
        p_param = {'name': 'MC_{}RATE_P'.format(axis.upper()), 'value': p}
        i_param = {'name': 'MC_{}RATE_I'.format(axis.upper()), 'value': i}
        d_param = {'name': 'MC_{}RATE_D'.format(axis.upper()), 'value': d}
        gains = {}
        for param in [p_param, i_param, d_param]:
            if param['value'] is not None:
                param_value = ParamValue()
                param_value.real = float(param['value'])
                response = self.set_param_service(param_id=param['name'], value=param_value)
                success = response.success
                value = response.value.real
                if success:
                    gains[param['name']] = value.real
                else:
                    gains[param['name']] = 'ERROR'
        return gains


class Procedures(PX4):

    def __init__(self):
        super().__init__()
        self.quadcopter_state = QuadcopterState()
        self.quadcopter_control = QuadcopterCommand()

    def arm_set_offboard(self):
        rospy.loginfo('Waiting for connection...')
        while not self.quadcopter_state.current_state().connected:
            self.rate.sleep()
        rospy.loginfo('Connection established!')

        rospy.loginfo('Sending initial position points...')
        start_quad_pose = self.quadcopter_state.start_pose()
        while start_quad_pose is None:
            start_quad_pose = self.quadcopter_state.start_pose()
        start_x = start_quad_pose.pose.position.x
        start_y = start_quad_pose.pose.position.y
        start_z = start_quad_pose.pose.position.z
        for i in range(0, 100):
            self.quadcopter_control.publish_pose(x=start_x, y=start_y, z=start_z)
            self.rate.sleep()
        rospy.loginfo('Initial points sent!')

        mode_resp = self.quadcopter_control.set_mode('OFFBOARD')
        while not mode_resp:
            mode_resp = self.quadcopter_control.set_mode('OFFBOARD')
        rospy.loginfo('Offboard set')
        time.sleep(1)
        arm_resp = self.quadcopter_control.arm(True)
        while not arm_resp:
            arm_resp = self.quadcopter_control.arm(True)
        rospy.loginfo('Armed')
