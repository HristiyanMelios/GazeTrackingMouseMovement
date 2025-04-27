import cv2
import numpy as np
import os

class Canonize():
    def __init__(self, mediapipe_landmarks=None, frame=None, intrinsic_matrix=None, distortion_coeff=None):
        if mediapipe_landmarks is None or frame is None:
            # Mark as uninitialized
            self.landmarks = None
            self.frame_w = None
            self.frame_h = None
            self.k = None
            self.dist = None
            self.model_points = None
            self.image_points = None
            return  # Don't crash, just skip full init


        landmark_dict = {
            "nose_tip": 1,
            "chin": 152,
            "left_eye_outer": 33,
            "left_eye_inner": 133,
            "right_eye_outer": 263,
            "right_eye_inner": 362,
            "left_mouth_corner": 61,
            "right_mouth_corner": 291,
        }

        # Passing an exisiting intriniscs matrix for testing purposes. Replace later with an actual method to fetch from user
        intrinsics = {}
        if intrinsic_matrix is None:
            for root, dir, files in os.walk('.'):
                for directory in dir:    
                    if directory == 'calib_imgs_intrinsics':
                        intrinsics = np.load(f"{directory}/intrinsics.npz")
                        break
            self.k = intrinsics['K']
            self.dist = intrinsics['dist']
        else:
            self.k = intrinsic_matrix
            self.dist = distortion_coeff

        # Obtaining landmarks from media_pipe. Will extract x,y components later using self._pt()
        self.landmarks = mediapipe_landmarks

        self.frame_w = frame.shape[0]
        self.frame_h = frame.shape[1]


        # Defining key points on the human face into a canonical pose
            # nose will be at origin and eyes are ~60 units apart from each other
            # Pupil's are about ~ 30 units away from the nose tip 
        self.model_points = np.array([
            (0.0, 0.0, 0.0), #nose
            (-60.0, 30.0, -10.0), #left eye outer corner
            (-30.0, 30.0, -10.0), #left eye inner corner
            (60.0, 30.0, -10.0), #right eye outer corner
            (30.0, 30.0, -10.0), #right eye inner corner
            (0.0, -60.0, -5.0) #chin
            
        ])

        self.image_points = np.array([
            self._pt(landmark_dict['nose_tip']),
            self._pt(landmark_dict['left_eye_outer']),
            self._pt(landmark_dict['left_eye_inner']),
            self._pt(landmark_dict['right_eye_outer']),
            self._pt(landmark_dict['right_eye_inner']),
            self._pt(landmark_dict['chin']),            
        ])

    @classmethod
    def empty_canonizer(cls):
        dummy_landmarks = None
        dummy_frame = None
        return cls(dummy_landmarks, dummy_frame)
    
    def get_img_pts(self):
        return self.image_points

    def mediapipe_landmarks():
        pass

    def _pt(self, index):
        """Fetches the landmark points at the given index as (x_px, y_px)"""
        lm = self.landmarks.landmark[index]
        return (int(lm.x * self.frame_w),
                int(lm.y * self.frame_h))

