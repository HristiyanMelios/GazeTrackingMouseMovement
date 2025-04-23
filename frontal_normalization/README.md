## Frontal Normalization

Attempting to account for the problem of the inaccuracies obtained from a head that's angled away from the camera. Done by finding a transformation function `M(R,t)` such that given an `(x,y)` coordinates of landmarks it transforms them as if the head is facing straight on the camera.

**Steps**
* Obtain a set of 3D templates a priori to calibration
* Obtain Camera Intrinsics of webcam- requires calibration using Checkerboard patterns or april tags.
* SolvePnP using the output of the above two. Solution will give us `M(R,t)`