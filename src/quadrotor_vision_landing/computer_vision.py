import cv2


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
        return bbox
