from __future__ import division
import cv2
import mediapipe as mp
import math
import numpy as np
from .eye import Eye
from .calibration import Calibration


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        # Initialize MediaPipe FaceMesh
        mp_face = mp.solutions.face_mesh
        self._face_mesh = mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        if frame is None or frame.size == 0:
            print('No frame detected')
            return

        results = self._face_mesh.process(frame)
        if not results.multi_face_landmarks:
            return

        # Pass landmarks into Eye
        try:
            mp_landmarks = results.multi_face_landmarks[0]
            self.eye_left = Eye(self.frame, mp_landmarks, 0,
                                self.calibration)
            self.eye_right = Eye(self.frame, mp_landmarks, 1,
                                 self.calibration)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        Takes each eye ratio separately, and returns if the ratio
        is within range, and also not noisy (such as only returning left
        eye when looking at the bottom right)
        """
        if not self.pupils_located:
            return None

        x_l, _ = self.pupil_left_coords()
        x_r, _ = self.pupil_right_coords()
        
        xs_l = [pt[0] for pt in self.eye_left.landmark_points]
        xl_min, xl_max = min(xs_l), max(xs_l)

        xs_r = [pt[0] for pt in self.eye_right.landmark_points]
        xr_min, xr_max = min(xs_r), max(xs_r)

        ratios = []
        if xl_max > xl_min:
            ratios.append((x_l - xl_min) / (xl_max - xl_min))
        if xr_max > xr_min:
            ratios.append((x_r - xr_min) / (xr_max - xr_min))

        if not ratios:
            return None

        hr = sum(ratios) / len(ratios)
        return min(max(hr, 0.0), 1.0)

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
            horizontal direction of the gaze. The extreme top is 0.0,
            the center is 0.5 and the extreme bottom is 1.0
            Takes an average of the combined eye ratios, and removes
            eye-roll by converting the angle of the eye corner and
            rotating the pupil and eyelid back into horizontal space
            """
        if not self.pupils_located:
            return None

        # get raw points
        (x_l, y_l), (x_r, y_r) = self.pupil_left_coords(), self.pupil_right_coords()
        pts_l = self.eye_left.landmark_points
        pts_r = self.eye_right.landmark_points

        def single_eye_ratio(pupil, landmarks):
            # choose corner pts
            left_corner = landmarks[0]
            right_corner = landmarks[1]
            dx, dy = right_corner - left_corner
            theta = math.atan2(dy, dx)  # compute roll angle
            # build rotation matrix
            R = np.array([[math.cos(-theta), -math.sin(-theta)],
                          [math.sin(-theta),  math.cos(-theta)]])
            # rotate all Y coords
            pup_rot = R.dot(np.array(pupil) - left_corner)
            lids_y = [R.dot(pt - left_corner)[1] for pt in landmarks[2:] ]
            top_rot, bottom_rot = min(lids_y), max(lids_y)
            # normalized vertical position
            return (pup_rot[1] - top_rot) / (bottom_rot - top_rot)

        vr_l = single_eye_ratio((x_l, y_l), pts_l)
        vr_r = single_eye_ratio((x_r, y_r), pts_r)
        vr = (vr_l + vr_r) / 2.0

        return float(np.clip(vr, 0.0, 1.0))

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= .45

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.55

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 5.68

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame
