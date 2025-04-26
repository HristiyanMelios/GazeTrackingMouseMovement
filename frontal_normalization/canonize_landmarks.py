import cv2
import numpy as np

class Canonize():
    def __init__(self, intrinsic_matrix, distortion_coeff):
        self.k = intrinsic_matrix
        self.dist = distortion_coeff

        # Defining key points on the human face into a canonical pose
        # nose will be at origin and eyes are ~60 units apart from each other
        # Pupil's are about ~ 30 units away from the nose tip 
        self.model_points = np.array([
            (0.0, 0.0, 0.0), #nose
            (-30.0, 30.0, -10.0), #left eye outer corner
            (30.0, 30.0, -10.0) #right eye outer corner
        ])

    def mediapipe_landmarks():
        pass