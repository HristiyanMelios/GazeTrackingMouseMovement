"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import numpy as np

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

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

        # dynamic scale so inner eye corners are ~100 px apart
        left_inner  = canon_pts[2]
        right_inner = canon_pts[4]
        eye_dist_mm = np.linalg.norm(left_inner - right_inner) + 1e-6
        scale = 100.0 / eye_dist_mm

        for (x, y, z) in canon_pts:
            # shift nose to origin, then scale, then center on canvas
            vis_x = int((x - nose_x) * scale + canvas_w / 2)
            vis_y = int((nose_y - y) * scale + canvas_h / 2)
            cv2.circle(canvas, (vis_x, vis_y), 4, (0, 0, 255), -1)

    # show both windows
    cv2.imshow('Canonicalized Landmarks', canvas)
    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
