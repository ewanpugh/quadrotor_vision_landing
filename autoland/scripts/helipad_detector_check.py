#!/home/njnano1/python36_ws/py36env/bin/python3.6

import rospy
from quadrotor_vision_landing.px4 import PX4
from quadrotor_vision_landing.computer_vision import ComputerVision
import time
import math
import cv2

if __name__ == '__main__':

    rospy.loginfo('Initialising')
    last_timestamp = 0
    px4 = PX4()
    quad_state = px4.quadcopter_state()
    quad_control = px4.quadcopter_control()
    computer_vision = ComputerVision()
    cascade = cv2.CascadeClassifier('/home/njnano1/python36_ws/src/autoland/detection_model/cascade.xml')

    while True:
        img = computer_vision.get_rgb_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        helipads = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(100, 100))
        for (x,y,w,h) in helipads:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        cv2.imshow('img',img)
        cv2.waitKey(1) 
