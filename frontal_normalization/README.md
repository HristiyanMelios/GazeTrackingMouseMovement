## Frontal Normalization

Attempting to account for the problem of the inaccuracies obtained from a head that's angled away from the camera. Done by finding a transformation function `M(R,t)` such that given an `(x,y)` coordinates of landmarks it transforms them as if the head is facing straight on the camera.

**Pre-req**:
* The camera's intrinsic's must be calculated for every new device this app is ran. get_intrinsics.py accomplishes this by forcing the user to load an 8x8 checkerboard pattern on their device or on a piece of paper and show it infront of the webcam. This process is very fast and only needs to be done once.

**Steps**
* Obtain a set of 3D templates a priori to calibration
* Obtain Camera Intrinsics of webcam- requires calibration using Checkerboard patterns or april tags.
* SolvePnP using the output of the above two. Solution will give us `M(R,t)`