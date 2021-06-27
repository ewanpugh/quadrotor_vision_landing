#!/home/vslam/python36_ws/py36env/bin/python3.6

import rospy
from quadrotor_vision_landing.px4 import PX4
import time
import math

if __name__ == '__main__':
    try:
        px4 = PX4()
        quad_state = px4.quadcopter_state()
        quad_control = px4.quadcopter_control()

        rospy.loginfo('Waiting for offboard mode')
        while quad_state.current_state().mode.lower() != 'offboard':
            quad_control.publish_velocity() # If empty commands zero

        rospy.loginfo('Waiting for 5 seconds')

        start_time = time.time()
        while time.time() - start_time < 5:
            quad_control.publish_velocity()

        for speed in [0.1]:
            for setpoint in [speed, -1*speed]:
                rospy.loginfo('Z setpoint = {}'.format(setpoint))
                start_time = time.time()
                while time.time() - start_time < 5:
                    quad_control.publish_velocity(z=setpoint)
                start_time = time.time()
                while time.time() - start_time < 1:
                    quad_control.publish_velocity(x=0)
        
        rospy.loginfo('Script ended, take control')
        while quad_state.current_state().mode.lower() == 'offboard': 
            quad_control.publish_velocity()
        
    except rospy.ROSInterruptException:
        pass
