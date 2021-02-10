import cv2
import numpy as np
import airsim
import time


class ComputerVision:

    @staticmethod
    def geometry_helipad_detection(im):
        orig = im.copy()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        circles = []
        h_list = []
        for contour in contours:
            try:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) >= 8 | len(approx) <= 16:
                    ((x, y), (h, w), _) = cv2.minAreaRect(contour)
                    ar = w / float(h)
                    if (ar > 0.75) & (ar < 0.85):
                        h_list.append({'x': x, 'y': y, 'w': w, 'h': h})
                        rect = cv2.minAreaRect(contour)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        orig = cv2.drawContours(orig, [box], 0, (255, 0, 0), 1)
                    else:
                        ellipse = cv2.fitEllipse(contour)
                        x, y = ellipse[0]
                        w = ellipse[1][0]
                        h = ellipse[1][1]
                        if (w/h > 0.95) & (w/h < 1.05):
                            circles.append({'x': x, 'y': y, 'w': w, 'h': h})
            except:
                pass

        helipad_detected = False
        helipad_centre = None
        for H in h_list:
            circle_same_center_as_h_count = 0
            for circle in circles:
                x_diff = abs(circle['x'] - H['x'])
                y_diff = abs(circle['y'] - H['y'])
                if (x_diff < 2) & (y_diff < 2):
                    circle_same_center_as_h_count += 1
            if circle_same_center_as_h_count == 2:
                helipad_centre = (circle['x'], circle['y'])
                helipad_detected = True

        return orig, helipad_detected, helipad_centre


client = airsim.MultirotorClient()
helipad_detector = ComputerVision()
iteration = 1
elapsed_detection_time = 0
number_frames_to_run = 10
while iteration <= number_frames_to_run:
    print(iteration)
    responses = client.simGetImages([airsim.ImageRequest("bottom_center", airsim.ImageType.Scene, False, False)])
    response = responses[0]

    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

    img_rgb = img1d.reshape(response.height, response.width, 3)

    start = time.time()
    img, detected_boolean, helipad_centroid = helipad_detector.geometry_helipad_detection(img_rgb)
    elapsed_detection_time += (time.time() - start)
    print('\t{}'.format(detected_boolean))
    print('\t{}'.format(helipad_centroid))
    iteration += 1


print('\nDetection time = {}s'.format(elapsed_detection_time/number_frames_to_run))
