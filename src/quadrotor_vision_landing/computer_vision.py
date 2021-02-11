import cv2
import numpy as np
import airsim
import time


class ComputerVision:

    def __init__(self):
        self.tracker = cv2.TrackerCSRT_create()
        self.ok = None

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
                        bounding_box = cv2.boundingRect(contour)
                        h_list.append({'x': x, 'y': y, 'w': w, 'h': h, 'bounding_box': bounding_box})
                    else:
                        ellipse = cv2.fitEllipse(contour)
                        x, y = ellipse[0]
                        w = ellipse[1][0]
                        h = ellipse[1][1]
                        if (w/h > 0.95) & (w/h < 1.05):
                            circles.append({'x': x, 'y': y, 'w': w, 'h': h})
            except:
                pass

        for H in h_list:
            for circle in circles:
                x_diff = abs(circle['x'] - H['x'])
                y_diff = abs(circle['y'] - H['y'])
                if (x_diff < 2) & (y_diff < 2):
                    helipad_centre = (circle['x'], circle['y'])
                    helipad_bounding_box = H['bounding_box']
                    return True, helipad_centre, helipad_bounding_box

        return False, None, None

    def init_tracker(self, frame, bounding_box):
        self.tracker.init(frame, bounding_box)

    def track_object(self, frame):

        ok, bbox = self.tracker.update(frame)

        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.imshow('Helipad detection', frame)
        cv2.waitKey(1)


client = airsim.MultirotorClient()
helipad_detector = ComputerVision()

initiate_tracker = False
tracker_initiated = False
cross_val_frames = 10  # Cross validate tracker with detection every 10 frames
tracker_frames = 0
while True:
    responses = client.simGetImages([airsim.ImageRequest("bottom_center", airsim.ImageType.Scene, False, False)])
    response = responses[0]

    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

    img_rgb = img1d.reshape(response.height, response.width, 3)

    if not tracker_initiated:
        detected_boolean, helipad_centroid, bb = helipad_detector.geometry_helipad_detection(img_rgb)
        print(detected_boolean)
        if detected_boolean:
            initiate_tracker = True
            if initiate_tracker & (not tracker_initiated):
                helipad_detector.init_tracker(img_rgb, bb)
                tracker_initiated = True
    else:
        if tracker_frames % cross_val_frames != 0:
            helipad_detector.track_object(img_rgb)
        else:
            detected_boolean, helipad_centroid, bb = helipad_detector.geometry_helipad_detection(img_rgb)
            if detected_boolean:
                helipad_detector.init_tracker(img_rgb, bb)
            helipad_detector.track_object(img_rgb)
        tracker_frames += 1
