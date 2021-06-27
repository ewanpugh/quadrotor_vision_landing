#!/home/njnano1/python36_ws/py36env/bin/python3.6

import rospy
from geometry_msgs.msg import Twist
import time

desired_velocity = None
last_recieved_time = None

def desired_velocity_cb(msg):
    global desired_velocity, last_recieved_time
    desired_velocity = msg
    last_recieved_time = time.time()

if __name__ == '__main__':
    rospy.init_node('autoland_desried_vel_node', anonymous=True)

    desired_velocity_sub = rospy.Subscriber('/autoland/desired_velocity', Twist, desired_velocity_cb)
    velocity_publisher = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=1)

    while True:
        start = time.time()
        if (desired_velocity is not None) & (last_recieved_time is not None):
            if (time.time() - last_recieved_time) < 2:
                velocity_publisher.publish(desired_velocity)
        while time.time() - start < 0.1: # Limit to 10 Hz
            pass
