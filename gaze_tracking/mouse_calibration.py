import cv2
import numpy as np
from gaze_tracking import GazeTracking
import time
import pyautogui
import statistics


class MouseCalibration:
    """
    Calibrates eye position to (x,y) monitor coordinates by collecting
    gaze ratios, and performing least squares regression
    """
    def __init__(self, tracker):
        """
        tracker: GazeTracker object
        screen_w and screen_h: Screen resolution taken from pyautogui
        """
        self.tracker = tracker

        # get monitor resolution
        self.screen_w, self.screen_h = pyautogui.size()
        W, H = self.screen_w, self.screen_h  # to reduce clutter in array
        self.targets = [
            # Targets will show starting from top left, going clockwise
            (0, 0), (W//2, 0),
            (W, 0), (W, H//2),
            (W, H), (W//2, H),
            (0, H), (0, H//2),
            (W//2, H//2),
        ]

        # Array to store the coordinate mappings when calibrating
        self.calibration_data = []

        self.skip_frames = 10
        self.capture_frames = 50
        self.window_name = 'Coordinate Calibration'
        self.canvas = np.zeros((self.screen_h, self.screen_w, 3), np.uint8)
        self.cap = None

    @staticmethod
    def _reject_outliers(data, m=1.5):
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower = q1 - m * iqr
        upper = q3 + m * iqr
        return [x for x in data if lower <= x <= upper]

    def _draw(self, point=None, text=None):
        """
        Simple function to create a new window and show text and calib points.
        """
        self.canvas.fill(0)

        if point is not None:
            cv2.circle(self.canvas, point, 20, (255, 255, 255), -1)

        if text is not None:
            font = cv2.FONT_HERSHEY_DUPLEX
            size = cv2.getTextSize(text, font, 1, 2)[0]
            pos = (self.screen_w//2 - size[0]//2,
                   self.screen_h//2 + size[1]//2)
            cv2.putText(self.canvas, text, pos, font, 1.0, (255, 255, 255), 2)

        cv2.imshow(self.window_name, self.canvas)
        cv2.waitKey(1)

    def calibrate(self, cam):
        """
        Calibrate eye position to (x,y) monitor coordinates by collecting
        gaze ratios, and performing least squares regression
        """
        self.cap = cam
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, 0, 0)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

        # Start calibration
        while True:
            self._draw(text="Press SPACE to start calibration")
            if cv2.waitKey(0) == 32:
                break

        # Countdown
        for i in (3,2,1):
            self._draw(text=str(i))
            time.sleep(1)

        # Draw each target, skip initial frames and get data frames
        for pt in self.targets:
            # Skip first few frames to settle eyes on point
            for skip_frame in range(self.skip_frames):
                ret, frame = self.cap.read()
                self.tracker.refresh(frame)
                self._draw(point=pt)

            # Capture the next frames within the arrays
            Ex, Ey = [], []
            for cap_frame in range(self.capture_frames):
                ret, frame = self.cap.read()
                self.tracker.refresh(frame)
                if self.tracker.pupils_located:
                    Ex.append(self.tracker.horizontal_ratio())
                    Ey.append(self.tracker.vertical_ratio())

                self._draw(point=pt)

            # Store average of data
            # Ex_avg = sum(Ex)/len(Ex)
            # Ey_avg = sum(Ey)/len(Ey)
            Ex_filtered = self._reject_outliers(Ex)
            Ey_filtered = self._reject_outliers(Ey)

            # Try median instead
            Ex_med = statistics.median(Ex_filtered)
            Ey_med = statistics.median(Ey_filtered)
            self.calibration_data.append(
                ((Ex_med,Ey_med), pt)
            )

        cv2.destroyWindow(self.window_name)

    def fit(self):
        """
        Least squares regression fit from Eye (Ex,Ey) to monitor coordinates (x,y)
        Returns six parameters: a, b, tx, c, d, ty.
        a,b,c,d are the shifts in eye ratios to map into movement
        tx and ty are the offsets for gaze_ratio calculation to stay centered
        """
        # 1D MAPPING SOLUTION
        # build vectors
        exs = np.array([ex for (ex, ey), _ in self.calibration_data])
        X_true = np.array([X for _, (X, _) in self.calibration_data])
        eys = np.array([ey for (ex, ey), _ in self.calibration_data])
        Y_true = np.array([Y for _, (_, Y) in self.calibration_data])

        # 1D least‐squares: X = a_h*ex + tx_h
        E_h = np.vstack([exs, np.ones_like(exs)]).T
        (a_h, tx_h), *_ = np.linalg.lstsq(E_h, X_true, rcond=None)

        # 1D least‐squares: Y = a_v*ey + ty_v
        E_v = np.vstack([eys, np.ones_like(eys)]).T
        (a_v, ty_v), *_ = np.linalg.lstsq(E_v, Y_true, rcond=None)

        return a_h, tx_h, a_v, ty_v
