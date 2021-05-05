import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')

import cv2
import numpy as np

rgb = None
depth = None
pointcloud = None

def _rgb_cb(image_data):
    global rgb
    rgb = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)


def _depth_cb(image_data):
    global depth
    depth = np.frombuffer(image_data.data, dtype=np.uint16).reshape(image_data.height, image_data.width, -1)


class ComputerVision:

    def __init__(self):
        self.tracker = None
        self.ok = None
        self.light_orange = (10, 100, 20)
        self.dark_orange = (25, 255, 255)
        self.rgb_sub = rospy.Subscriber('/camera/color/image_raw', Image, _rgb_cb)
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, _depth_cb)
        self.bb_publisher = rospy.Publisher('autoland/helipad/bb', String, queue_size=10)

    def publish_bounding_box(self, bb):
        x, y, w, h = bb
        dictionary = {"x": x, "y": y, "w": w, "h": h}
        self.bb_publisher.publish(str(dictionary))

    @staticmethod
    def get_rgb_image():
        """
        Get an rgb image from the RealSense /camera/color/image_raw topic

        :returns:
            - RGB image (:py:class:`np.array`) - 3D np.array
        """
        global rgb
        while rgb is None:
            pass
        return rgb

    @staticmethod
    def get_depth_image():
        """
        Get a depth image from the RealSense /camera/aligned_depth_to_color/image_raw topic

        :returns:
            - depth image (:py:class:`np.array`) - 2D np.array containing depth data
        """
        global depth
        while depth is None:
            pass
        return depth

    def init_tracker(self, image, bounding_box):
        """
        Initiate an OpenCV CSRT tracker object

        :param image: RGB Image
        :type image: :py:class:`np.array`
        :param bounding_box: OpenCV bounding box of the object to track in the image
        :type bounding_box: :py:class:`list`
        """
        self.__dict__.pop('tracker')
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(image, bounding_box)

    def track_object(self, image):
        """
        Get the bounding box of the object from the tracker.
        :meth:`quadrotor_vision_landing.ComputerVision.init_tracker` must be initialised first!

        :param image: RGB Image
        :type image: :py:class:`np.array`
        :returns:
            bounding box (:py:class:`list`) - OpenCV bounding box of the object
        """
        ok, bbox = self.tracker.update(image)
        return bbox

    def geometry_helipad_detection(self, image):
        """
        Get the location of the helipad im the image using geometry approach

        :param image: RGB Image
        :type image: 3D np.array

        :returns:
            - **detected boolean** (:py:class:`bool`) - True if helipad detected otherwise False
            - **helipad centre** (:py:class:`tuple`) - Tuple containing x and y location of helipad centre
            - **helipad bounding box** (:py:class:`list`) - OpenCV bounding box containing the helipad

        """
        im = cv2.bilateralFilter(image, 9, 75, 75)
        im_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

        mask = cv2.inRange(im_hsv, self.light_orange, self.dark_orange)

        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]	

        circles = []
        h_list = []
        i = 1
        for contour in contours:
            try:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) >= 8 | len(approx) <= 16:
                    ((x, y), (h, w), _) = cv2.minAreaRect(contour)
                    ar = w / float(h)

                    if ((w / float(h) > 0.7) & (w / float(h) < 0.8)) | ((h / float(w) > 0.7) & (h / float(w) < 0.8)):
                        bounding_box = cv2.boundingRect(contour)
                        h_list.append({'x': x, 'y': y, 'w': w, 'h': h, 'bounding_box': bounding_box})
                    else:
                        ellipse = cv2.fitEllipse(contour)
                        x, y = ellipse[0]
                        w = ellipse[1][0]
                        h = ellipse[1][1]
                        if (w/h > 0.9) & (w/h < 1.1):
                            circles.append({'x': x, 'y': y, 'w': w, 'h': h})
            except:
                pass
        for H in h_list:
            for circle in circles:
                x_diff = abs(circle['x'] - H['x'])
                y_diff = abs(circle['y'] - H['y'])
                if (x_diff < 2) & (y_diff < 2):
                    helipad_centre = (int(circle['x']), int(circle['y']))
                    helipad_bounding_box = H['bounding_box']
                    return True, helipad_centre, helipad_bounding_box

        return False, (), []
