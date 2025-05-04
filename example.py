"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import numpy as np

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)


scale_fixed     = True   # set once and keep forever
smoothed_scale  = None   # exponentially–smoothed per frame (option B)
α_scale         = 0.05   # smoothing coefficient for option B

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

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

    cv2.putText(frame, text, (90, 60),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # get the stabilized, canonicalized 3D landmarks
    canon_pts = gaze.get_canonized_lm()
    # prepare a 500×500 white canvas
    canvas_h, canvas_w = 500, 500
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    if canon_pts.shape[0] >= 6:
        # nose is first point
        nose_x, nose_y, _ = canon_pts[0]

        # current inner‑eye distance in canonical millimetres
        eye_dist_mm = np.linalg.norm(canon_pts[2] - canon_pts[4]) + 1e-6

        # ───── Option A: freeze scale after first frame ─────
        if scale_fixed is None:
            scale_fixed = 100.0 / eye_dist_mm    # set once
        scale = scale_fixed

        # ───── Option B: low‑pass the scale (comment out if using A) ─────
        if smoothed_scale is None:
            smoothed_scale = 100.0 / eye_dist_mm
        else:
            target_scale   = 100.0 / eye_dist_mm
            smoothed_scale = α_scale * target_scale + (1 - α_scale) * smoothed_scale
        scale = smoothed_scale
        # ─────────────────────────────────────────────────────────────────

        # draw the landmarks
        for (x, y, z) in canon_pts:
            vis_x = int((x - nose_x) * scale + canvas_w / 2)
            vis_y = int((nose_y - y) * scale + canvas_h / 2)
            cv2.circle(canvas, (vis_x, vis_y), 4, (0, 0, 255), -1)

        # quick sanity read‑out
        print(f"eye‑eye dist canonical: {eye_dist_mm:.2f} mm")

    # show both windows
    cv2.imshow('Canonicalized Landmarks', canvas)
    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
