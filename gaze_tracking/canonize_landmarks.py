import cv2
import numpy as np
import os

class Canonize():
    def __init__(
        self,
        mediapipe_landmarks=None,
        frame=None,
        intrinsic_matrix=None,
        distortion_coeff=None,
        smoothing_alpha=0.2
    ):
        # Empty state if no initial landmarks/frame
        if mediapipe_landmarks is None or frame is None:
            self.landmarks = None
            self.frame_w = None
            self.frame_h = None
            self.k = None
            self.dist = None
            self.model_points = None
            self.image_points = None
            self.prev_rvec = None
            self.prev_tvec = None
            self.prev_canon = None
            self.smoothing_alpha = smoothing_alpha
            return

        # Initialize intrinsics
        if intrinsic_matrix is None:
            for root, dirs, files in os.walk('.'):
                if 'calib_imgs_intrinsics' in dirs:
                    intr = np.load(os.path.join(root, 'calib_imgs_intrinsics', 'intrinsics.npz'))
                    self.k = intr['K']
                    self.dist = intr['dist']
                    break
        else:
            self.k = intrinsic_matrix
            self.dist = distortion_coeff

        self.landmarks = mediapipe_landmarks
        self.frame_w = frame.shape[1]
        self.frame_h = frame.shape[0]
        self.smoothing_alpha = smoothing_alpha

        # Default 3D model points
        self.model_points = np.array([
            (0.0,   0.0,   0.0),    # Nose tip
            (-60.0, 30.0, -10.0),    # Left eye outer
            (-30.0, 30.0, -10.0),    # Left eye inner
            (60.0,  30.0, -10.0),    # Right eye outer
            (30.0,  30.0, -10.0),    # Right eye inner
            (0.0,  -60.0,  -5.0)     # Chin
        ], dtype=np.float32)

        # Previous state for smoothing
        self.prev_rvec = None
        self.prev_tvec = None
        self.prev_canon = None

    @classmethod
    def empty_canonizer(cls):
        return cls(None, None, None, None)

    def refresh(self, mediapipe_landmarks, frame):
        # On first real call, ensure defaults exist
        if self.k is None:
            # Reinitialize by creating a fresh instance
            tmp = Canonize(mediapipe_landmarks, frame)
            # Copy over defaults
            self.k = tmp.k
            self.dist = tmp.dist
            self.model_points = tmp.model_points
            self.smoothing_alpha = tmp.smoothing_alpha
            self.prev_rvec = tmp.prev_rvec
            self.prev_tvec = tmp.prev_tvec
            self.prev_canon = tmp.prev_canon

        self.landmarks = mediapipe_landmarks
        self.frame_w = frame.shape[1]
        self.frame_h = frame.shape[0]

        # Build image_points
        ld = self.landmarks.landmark
        idxs = [1, 33, 133, 263, 362, 152]
        pts = [(ld[i].x * self.frame_w, ld[i].y * self.frame_h) for i in idxs]
        self.image_points = np.array(pts, dtype=np.float32).reshape(-1, 2)

        if self.image_points.shape[0] < 4:
            return np.zeros((0, 3), dtype=np.float32)

        # SolvePnP with extrinsic guess
        flags = cv2.SOLVEPNP_ITERATIVE
        if self.prev_rvec is not None and self.prev_tvec is not None:
            success, rvec, tvec = cv2.solvePnP(
                self.model_points, self.image_points,
                self.k, self.dist,
                self.prev_rvec, self.prev_tvec,
                True, flags
            )
        else:
            success, rvec, tvec = cv2.solvePnP(
                self.model_points, self.image_points,
                self.k, self.dist,
                flags=flags
            )
        if not success:
            return np.zeros((0, 3), dtype=np.float32)

        # Smooth rvec/tvec
        if self.prev_rvec is None:
            self.prev_rvec, self.prev_tvec = rvec, tvec
        else:
            α = self.smoothing_alpha
            self.prev_rvec = α * rvec + (1 - α) * self.prev_rvec
            self.prev_tvec = α * tvec + (1 - α) * self.prev_tvec

        # Convert to rotation matrix
        R, _ = cv2.Rodrigues(self.prev_rvec)
        t = self.prev_tvec

        # Depth estimation via median
        # Zs = R[2] @ self.model_points.T + t[2, 0]
        # Z_est = float(np.median(Zs))

        # Back-project + canonize
        canon = []
        for (u, v) in self.image_points:
            Zs = (R @ self.model_points.T + t).T[:, 2]        # shape (6,)
            Z_est = float(np.median(Zs))
            x_cam = (u - self.k[0, 2]) / self.k[0, 0] * Z_est
            y_cam = (v - self.k[1, 2]) / self.k[1, 1] * Z_est
            X_cam = np.array([x_cam, y_cam, Z_est], dtype=np.float32).reshape(3, 1)
            X_obj = R.T @ (X_cam - t)
            canon.append(X_obj[:, 0])
        canon_pts = np.vstack(canon)

        # Smooth canonical 2D points
        if self.prev_canon is None:
            self.prev_canon = canon_pts
        else:
            α = self.smoothing_alpha
            canon_pts = α * canon_pts + (1 - α) * self.prev_canon
            self.prev_canon = canon_pts

        return canon_pts

    def get_img_pts(self):
        return self.image_points
