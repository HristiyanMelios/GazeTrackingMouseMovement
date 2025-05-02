"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
from gaze_tracking.mouse_calibration import MouseCalibration
import pyautogui

gaze = GazeTracking()
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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
a_h, tx_h, a_v, ty_v = calibrator.fit()
print(f"Fitted affine params: a_h={a_h:.4f}, tx_h={tx_h:.4f}, a_v={a_v:.1f}, "
      f"ty_v={ty_v:.4f}")

# Evaluate residuals
errs = []
w, h = pyautogui.size()
print("\nCalibration residuals:")
for (ex, ey), (X_true, Y_true) in calibrator.calibration_data:
    X_pred = min(max((a_h * ex + tx_h), 0), w)  # clamp to [0, w]
    Y_pred = min(max((a_v * ey + ty_v), 0), h)  # clamp to [0, h]
    err = ((X_pred - X_true)**2 + (Y_pred - Y_true)**2)**0.5
    errs.append(err)
    print(f"  target: {(X_true, Y_true)} → pred: ({X_pred:.1f}, {Y_pred:.1f}), errr {err:.1f}px")

mean_err = sum(errs)/len(errs)
max_err = max(errs)
print(f"\nMean error: {mean_err:.1f}px,  Max error: {max_err:.1f}px\n")
print("Calibration complete. Entering live gaze demo")

# ───── End Calibration ────────────────────

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    if not _ or frame is None:
        print("Failed to grab frame")
        break  # or continue

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    # ex = gaze.horizontal_ratio()
    # ey = gaze.vertical_ratio()
    # X_pred = ex * w
    # Y_pred = ey * h
    # print(f"Raw map → ({X_pred:.0f}, {Y_pred:.0f}) vs. actual cursor")

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
