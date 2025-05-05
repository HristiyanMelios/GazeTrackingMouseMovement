"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
import pyautogui
import numpy as np
from gaze_tracking import GazeTracking
from gaze_tracking.mouse_calibration import MouseCalibration


gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    raise RuntimeError('Cannot open webcam')

calibrator = MouseCalibration(gaze)
print("Starting calibration... look at each dot in turn.")
calibrator.calibrate(webcam)

#  Fit affine mapping
calibrator.fit()
print("Calibration complete. Entering live gaze demo")
w, h = pyautogui.size()
prev_x, prev_y = pyautogui.position()
left_closed_prev = False
right_closed_prev = False
alpha = 0.2
smoothed_x, smoothed_y = prev_x, prev_y

while True:
    _, frame = webcam.read()

    if not _ or frame is None:
        print("⚠️ Failed to grab frame")
        break

    #  We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    #  Get your raw gaze ratios
    ex = gaze.horizontal_ratio()
    ey = gaze.vertical_ratio()

    if ex is not None and ey is not None:
        X_pred, Y_pred = calibrator._poly_model.predict([[ex, ey]])[0]

        #  Clamp to the monitor bounds
        X_clamped = np.clip(X_pred, 0, w)
        Y_clamped = np.clip(Y_pred, 0, h)

        smoothed_x = alpha * X_clamped + (1 - alpha) * smoothed_x
        smoothed_y = alpha * Y_clamped + (1 - alpha) * smoothed_y

        #  Delta‑guard to prevent spikes
        dx = smoothed_x - prev_x
        dy = smoothed_y - prev_y
        max_jump = 55  # pixels per frame
        X_final = prev_x + np.clip(dx, -max_jump, max_jump)
        Y_final = prev_y + np.clip(dy, -max_jump, max_jump)

        #  Move the OS cursor
        pyautogui.moveTo(int(X_final), int(Y_final))

        # Blink‑based clicking
        blink_thresh = 5.68
        left_closed = gaze.eye_left.blinking > blink_thresh
        right_closed = gaze.eye_right.blinking > blink_thresh

        if left_closed and not left_closed_prev:
            pyautogui.click(button='left')
        if right_closed and not right_closed_prev:
            pyautogui.click(button='right')

        # Save state for next frame
        left_closed_prev, right_closed_prev = left_closed, right_closed
        prev_x, prev_y = X_final, Y_final

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()
