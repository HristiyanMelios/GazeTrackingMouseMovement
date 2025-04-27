import math
import numpy as np
import cv2
from .pupil import Pupil


class Eye(object):
    """
    This class creates a new frame to isolate the eye and
    initiates the pupil detection.
    """

    LEFT_EYE_POINTS = [33, 133, 159, 145]
    RIGHT_EYE_POINTS = [362, 263, 386, 374]

    def __init__(self, original_frame, mp_landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None

        # store MediaPipe landmarks and frames
        self.landmarks = mp_landmarks
        h, w = original_frame.shape[:2]
        self.frame_h, self.frame_w = h, w

        self._analyze(original_frame, mp_landmarks, side, calibration)

    def _pt(self, index):
        """Fetches the landmark points at the given index as (x_px, y_px)"""
        lm = self.landmarks.landmark[index]
        return (int(lm.x * self.frame_w),
                int(lm.y * self.frame_h))

    @staticmethod
    def _middle_point(p1, p2):
        """Returns the middle point (x,y) between two points

        Arguments:
            p1 (mp.point): First point
            p2 (mp.point): Second point
        """
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """Isolate an eye, to have a frame without other part of the face.

        Arguments:
            frame (numpy.ndarray): Frame containing the face
            landmarks (self._pt): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)
        """
        region = np.array([self._pt(pt) for pt in points])
        region = region.astype(np.int32)
        self.landmark_points = region

        # Applying a mask to get only the eye
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # Cropping on the eye
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
        """Calculates a ratio that can indicate whether an eye is closed or not.
        It's the division of the width of the eye, by its height.

        Arguments:
            landmarks (mediapipe_landmarks): Facial landmarks for the face region
            points (list): Points of an eye (from the mediapipe landmark points in _pts)

        Returns:
            The computed ratio
        """
        left = self._pt(points[0])
        right = self._pt(points[1])
        top = self._pt(points[2])
        bottom = self._pt(points[3])

        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = float('inf') # Return large float to stop crashes when dividing by NoneType

        return ratio

    def _analyze(self, original_frame, landmarks, side, calibration):
        """Detects and isolates the eye in a new frame, sends data to the calibration
        and initializes Pupil object.

        Arguments:
            original_frame (numpy.ndarray): Frame passed by the user
            landmarks (self.<EYE>_POINTS): Facial landmarks for the face region
            side: Indicates whether it's the left eye (0) or the right eye (1)
            calibration (calibration.Calibration): Manages the binarization threshold value
        """

        if side == 0:
            pts = self.LEFT_EYE_POINTS
            iris_idx = (468, 469, 470, 471, 472)
        elif side == 1:
            pts = self.RIGHT_EYE_POINTS
            iris_idx = (473, 474, 475, 476, 477)
        else:

            return
        self.blinking = self._blinking_ratio(landmarks, pts)
        self._isolate(original_frame, self.landmarks, pts)

        # Pixel coordinates for iris landmarks
        iris_pts = [self._pt(i) for i in iris_idx]

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        # x and y average
        cx = sum(p[0] for p in iris_pts) / len(iris_pts)
        cy = sum(p[1] for p in iris_pts) / len(iris_pts)

        # local coordinates
        px = int(cx - self.origin[0])
        py = int(cy - self.origin[1])

        # Store pupil location
        class SimplePoint:
            def __init__(self, x, y):
                self.x, self.y = x, y

        self.pupil = SimplePoint(px, py)
