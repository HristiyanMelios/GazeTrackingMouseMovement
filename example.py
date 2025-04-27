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

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    img_pts = gaze.get_img_pts()
    # if len(img_pts) > 0:
    #     for (x, y, z) in img_pts:
    #         print(int(x*frame.shape[0]), int(y*frame.shape[1]))
    #         cv2.circle(frame, (int(x*frame.shape[1]), int(y*frame.shape[0])), 3, (0, 255, 0), -1)



    canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255  # white canvas

    for (x, y, z) in img_pts:
        # Scale and center for visualization
        vis_x = int(x * 2 + 250)
        vis_y = int(-y * 2 + 250)  # invert y-axis to match screen coordinates

        # Draw a small circle for each point
        cv2.circle(frame, (vis_x, vis_y), 4, (0, 0, 255), -1)  # Red filled dot

    # cv2.imshow('Canonicalized Landmarks', canvas)
    # cv2.waitKey(1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()
