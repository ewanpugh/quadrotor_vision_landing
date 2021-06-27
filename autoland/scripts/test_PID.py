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
        procedures = px4.procedures()

        procedures.arm_set_offboard()

        ### DEFAULT PID ###
        # {'MC_ROLLRATE_P': 0.20000000298023224,
        # 'MC_ROLLRATE_I': 0.20000000298023224,
        # 'MC_ROLLRATE_D': 0.003000000026077032}

        px4_pid = px4.pid()
        kp = 0.2
        ki = 0.2
        kd = 0.03
        px4_pid.set_gains('roll', p=kp, i=ki, d=kd)
        print(px4_pid.get_gains('roll'))

        desired_altitude = 5
        start_alt = quad_state.start_pose().pose.position.z
        current_alt = quad_state.current_pose().pose.position.z
        desired_vertical_vel = 1

        while current_alt - start_alt < desired_altitude:

            quad_control.publish_velocity(z=desired_vertical_vel)
            current_alt = quad_state.current_pose().pose.position.z

        rospy.loginfo('Taken off to {:.2f}m'.format(current_alt))
        start_time = time.time()

        for speed in range(1, 6):
            for setpoint in [speed, -1*speed]:
                rospy.loginfo('Setpoint = {}'.format(setpoint))
                start_time = time.time()
                while time.time() - start_time < 4:
                    quad_control.publish_velocity(x=setpoint)
                start_time = time.time()
                while time.time() - start_time < 2:
                    quad_control.publish_velocity(x=0)


    except rospy.ROSInterruptException:
        pass
