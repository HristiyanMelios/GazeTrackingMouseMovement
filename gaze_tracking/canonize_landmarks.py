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
        
        self.Z = np.array([])
        # Obtaining landmarks from media_pipe. Will extract x,y components later using self._pt()
        self.landmarks = mediapipe_landmarks

        self.frame_w = frame.shape[1]
        self.frame_h = frame.shape[0]

        # Defining key points on the human face into a canonical pose
            # nose will be at origin and eyes are ~60 units apart from each other
            # Pupil's are about ~ 30 units away from the nose tip 
        self.model_points = np.array([
            (0.0, 0.0, 0.0), #nose tip
            (-60.0, 30.0, -10.0), #left eye outer corner
            (-30.0, 30.0, -10.0), #left eye inner corner
            (60.0, 30.0, -10.0), #right eye outer corner
            (30.0, 30.0, -10.0), #right eye inner corner
            (0.0, -60.0, -5.0)     #chin  
        ])

        self.image_points = np.array([
            self._pt(landmark_dict['nose_tip']),
            self._pt(landmark_dict['left_eye_outer']),
            self._pt(landmark_dict['left_eye_inner']),
            self._pt(landmark_dict['right_eye_outer']),
            self._pt(landmark_dict['right_eye_inner']),
            self._pt(landmark_dict['chin']),            
        ], dtype=np.float32)

        self.R = np.ones((3,3))
        self.tvec = np.ones((3,1))

    @classmethod
    def empty_canonizer(cls):
        dummy_landmarks = None
        dummy_frame = None
        return cls(dummy_landmarks, dummy_frame)
    
    def get_img_pts(self):
        # return self.image_points
        return self._canonize()

    def solve_pnp(self):
        success, rvec, tvec = cv2.solvePnP(
            self.model_points, self.image_points, self.k, self.dist,
            flags = cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            self.R, _ = cv2.Rodrigues(rvec)
            self.tvec = tvec
            self.Z_est = (self.R[2] @ self.model_points[0]) + self.tvec[2,0]
        if not success:
            raise RuntimeError("solvePnP failed")
    
    # def _canonize(self):
    #     """Back-project each pixel into camera space using its true depth,
    #        then transform into the object frame."""
    #     canon = []
    #     # Compute Z_cam for each model point once:
    #     Zs = (self.R[2] @ self.model_points.T) + self.tvec[2,0]
    #     for (u, v), Z in zip(self.image_points, Zs):
    #         # Back project model points into camera frame
    #         x_cam = (u - self.k[0,2]) / self.k[0,0] * Z
    #         y_cam = (v - self.k[1,2]) / self.k[1,1] * Z
    #         X_cam = np.array([x_cam, y_cam, Z])

    #         # Canonize image points into object frame
    #         X_obj = self.R.T @ (X_cam - self.tvec)
    #         canon.append(X_obj[:2])  # we only need x,y
    #     return np.vstack(canon)    # shape = (n_landmarks, 2)
    def _canonize(self):
        canon = []
        
        # Estimate single global Z_cam using nose tip
        
        for (u, v) in self.image_points:
            # Back-project using estimated depth
            x_cam = (u - self.k[0,2]) / self.k[0,0] * self.Z_est
            y_cam = (v - self.k[1,2]) / self.k[1,1] * self.Z_est
            X_cam = np.array([x_cam, y_cam, self.Z_est])

            # Canonical transform
            X_obj = self.R.T @ (X_cam.reshape(3,1) - self.tvec)
            canon.append(X_obj[:,0])  # x,y,z

        return np.vstack(canon)
    
    def refresh(self, mediapipe_landmarks, frame):
        # Check if Canonizer was initialized empty
        if self.k is None or self.dist is None:
            # Try loading intrinsics if needed
            intrinsics = {}
            for root, dir, files in os.walk('.'):
                for directory in dir:
                    if directory == 'calib_imgs_intrinsics':
                        intrinsics = np.load(f"{directory}/intrinsics.npz")
                        break
            self.k = intrinsics['K']
            self.dist = intrinsics['dist']

        self.landmarks = mediapipe_landmarks
        self.frame_w = frame.shape[1]
        self.frame_h = frame.shape[0]

        # Reinitialize model points if they were empty
        if self.model_points is None:
            self.model_points = np.array([
                (0.0, 0.0, 0.0),  # nose tip
                (-60.0, 30.0, -10.0),  # left eye outer
                (-30.0, 30.0, -10.0),  # left eye inner
                (60.0, 30.0, -10.0),   # right eye outer
                (30.0, 30.0, -10.0),   # right eye inner
                (0.0, -60.0, -5.0)     # chin
            ], dtype=np.float32)

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

        self.image_points = np.array([
            self._pt(landmark_dict['nose_tip']),
            self._pt(landmark_dict['left_eye_outer']),
            self._pt(landmark_dict['left_eye_inner']),
            self._pt(landmark_dict['right_eye_outer']),
            self._pt(landmark_dict['right_eye_inner']),
            self._pt(landmark_dict['chin']),
        ], dtype=np.float32)

        self.image_points = self.image_points.reshape(-1, 2)

        # Sanity check before solvePnP
        if self.image_points.shape[0] < 4:
            print("[WARN] Not enough points to solvePnP. Skipping this frame.")
            return np.zeros((0, 3))

        self.solve_pnp()
        return self._canonize()

    # def refresh(self, mediapipe_landmarks, frame):
    #     self.landmarks = mediapipe_landmarks
    #     self.frame_w = frame.shape[1]
    #     self.frame_h = frame.shape[0]

    #     # Update image points
    #     landmark_dict = {
    #         "nose_tip": 1,
    #         "chin": 152,
    #         "left_eye_outer": 33,
    #         "left_eye_inner": 133,
    #         "right_eye_outer": 263,
    #         "right_eye_inner": 362,
    #         "left_mouth_corner": 61,
    #         "right_mouth_corner": 291,
    #     }

    #     self.image_points = np.array([
    #         self._pt(landmark_dict['nose_tip']),
    #         self._pt(landmark_dict['left_eye_outer']),
    #         self._pt(landmark_dict['left_eye_inner']),
    #         self._pt(landmark_dict['right_eye_outer']),
    #         self._pt(landmark_dict['right_eye_inner']),
    #         self._pt(landmark_dict['chin']),            
    #     ], dtype=np.float32)

    #     if self.image_points.shape[0] < 4:
    #         print("[WARN] Not enough points to solvePnP. Skipping this frame.")
    #         return np.zeros((0, 3))  # safe fallback

    #     self.solve_pnp() 
    #     return self._canonize()
        
    def _pt(self, index):
        """Fetches the landmark points at the given index as (x_px, y_px)"""
        lm = self.landmarks.landmark[index]
        return ((lm.x * self.frame_w),
                (lm.y * self.frame_h))

