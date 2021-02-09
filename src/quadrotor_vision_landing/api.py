import rospy
import time
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

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

    @classmethod
    def quadcopter_state(cls):
        return QuadcopterState()

    @classmethod
    def quadcopter_control(cls):
        return QuadcopterCommand()

    @classmethod
    def procedures(cls):
        return Procedures()


class QuadcopterState(PX4):

    def __init__(self):
        super().__init__()
        self.state_sub = rospy.Subscriber('mavros/state', State, state_cb)
        self.pose_publisher = rospy.Subscriber('mavros/local_position/pose', PoseStamped, pose_cb)
        self.local_vel_sub = rospy.Subscriber('/mavros/local_position/velocity_body', TwistStamped, vel_cb)

    @classmethod
    def current_state(cls):
        global current_state
        while current_state is None:
            pass
        return current_state

    @classmethod
    def current_pose(cls):
        global current_pose
        while current_pose is None:
            pass
        return current_pose

    @classmethod
    def start_pose(cls):
        global start_pose
        while start_pose is None:
            pass
        return start_pose

    @classmethod
    def current_velocity(cls):
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
        msg = PoseStamped()
        if x is not None:
            msg.pose.position.x = x
        if y is not None:
            msg.pose.position.y = y
        if z is not None:
            msg.pose.position.z = z
        if x_orient is not None:
            msg.pose.orientation.x = x_orient
        if y_orient is not None:
            msg.pose.orientation.y = y_orient
        if z_orient is not None:
            msg.pose.orientation.z = z_orient
        if w_orient is not None:
            msg.pose.orientation.w = w_orient

        self.pose_publisher.publish(msg)

    def publish_velocity(self, x=None, y=None, z=None, x_ang=None, y_ang=None, z_ang=None):
        msg = Twist()
        if x is not None:
            msg.linear.x = x
        if y is not None:
            msg.linear.y = y
        if z is not None:
            msg.linear.z = z
        if x_ang is not None:
            msg.angular.x = x_ang
        if y_ang is not None:
            msg.angular.y = y_ang
        if z_ang is not None:
            msg.angular.z = z_ang
        self.velocity_publisher.publish(msg)

    def set_mode(self, mode):
        return self.mode_service(custom_mode=mode)

    def arm(self, command):
        return self.arm_service(command)


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
