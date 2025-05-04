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


# ───── Calibration ────────────────────
calibrator = MouseCalibration(gaze)
print("Starting calibration... look at each dot in turn.")
calibrator.calibrate(webcam)

print("Raw calibration ratios:")
for (ex, ey), (X, Y) in calibrator.calibration_data:
    print(f"  target {(X,Y)} → ex={ex:.3f}, ey={ey:.3f}")
print("\n   ex range:",
      min(e for (e, _), _ in calibrator.calibration_data),
      max(e for (e, _), _ in calibrator.calibration_data))
print("\n   ey range:",
      min(e for (_, e), _ in calibrator.calibration_data),
      max(e for (_, e), _ in calibrator.calibration_data))

# Fit affine mapping
calibrator.fit()

# Evaluate residuals
errs = []
w, h = pyautogui.size()
print("\nCalibration residuals:")
for (ex, ey), (X_true, Y_true) in calibrator.calibration_data:
    X_pred, Y_pred = calibrator._poly_model.predict([[ex, ey]])[0]
    X_clamped = max(0, min(w, X_pred))
    Y_clamped = max(0, min(h, Y_pred))

    err = ((X_clamped - X_true)**2 + (Y_clamped - Y_true)**2)**0.5
    errs.append(err)
    print(f"  target: {(X_true, Y_true)} → pred: ({X_clamped:.1f}, {Y_clamped:.1f}), errr {err:.1f}px")

mean_err = sum(errs)/len(errs)
max_err = max(errs)
print(f"\nMean error: {mean_err:.1f}px,  Max error: {max_err:.1f}px\n")
print("Calibration complete. Entering live gaze demo")

# ───── End Calibration ────────────────────


prev_x, prev_y = pyautogui.position()
left_closed_prev = False
right_closed_prev = False
alpha = 0.2
smoothed_x, smoothed_y = prev_x, prev_y

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    if not _ or frame is None:
        print("⚠️ Failed to grab frame")
        break  # or continue

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    # 1) Get your raw gaze ratios
    ex = gaze.horizontal_ratio()
    ey = gaze.vertical_ratio()

    if ex is not None and ey is not None:
        # 2) Predict screen coords (use your affine or poly model)
        #    If you stuck with RANSAC, swap in calibrator._ransac_x/_ransac_y
        X_pred, Y_pred = calibrator._poly_model.predict([[ex, ey]])[0]

        # 3) Clamp to the monitor bounds
        X_clamped = np.clip(X_pred, 0, w)
        Y_clamped = np.clip(Y_pred, 0, h)

        smoothed_x = alpha * X_clamped + (1 - alpha) * smoothed_x
        smoothed_y = alpha * Y_clamped + (1 - alpha) * smoothed_y

        # 4) Delta‑guard to prevent spikes
        dx = smoothed_x - prev_x
        dy = smoothed_y - prev_y
        max_jump = 55  # pixels per frame
        X_final = prev_x + np.clip(dx, -max_jump, max_jump)
        Y_final = prev_y + np.clip(dy, -max_jump, max_jump)

        # 5) Move the OS cursor
        pyautogui.moveTo(int(X_final), int(Y_final))

        # 6) Blink‑based clicking
        #    Eye‑closure ratio lives in gaze.eye_left.blinking (and .eye_right)
        blink_thresh = 5.68
        left_closed = gaze.eye_left.blinking > blink_thresh
        right_closed = gaze.eye_right.blinking > blink_thresh

        if left_closed and not left_closed_prev:
            pyautogui.click(button='left')
        if right_closed and not right_closed_prev:
            pyautogui.click(button='right')

        # 7) Save state for next frame
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
