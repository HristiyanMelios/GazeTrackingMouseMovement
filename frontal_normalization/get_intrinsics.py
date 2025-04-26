import numpy as np
import cv2 as cv
import os

cap = cv.VideoCapture(0)
assert cap.isOpened(), "Could not open webcam"


rows, cols, square_size = 7, 7, 1.0

# Output dir. for images
out_dir = "calib_imgs_intrinsics"
os.makedirs(out_dir, exist_ok=True)

# Creating a grid of object points
pattern_size = (cols, rows)
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1,2) * square_size

# Storage
objpoints = []
imgpoints = []
saved_frames = 0
total_needed = 30


print("[INFO] Move the checkerboard around. Press q to stop.")

while saved_frames < total_needed:

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Can't receive frame (stream end). Exiting ...")
        break
    
    # Convert BGR to Gray
    gry = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Try to find checkerboard corners
    ret, corners = cv.findChessboardCorners(gry, pattern_size, None)
    print("[DEBUG] findChessboardCorners:", "FOUND" if ret else "NOT FOUND")

    # If able to find corners then get object points and image points

    if ret:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv.cornerSubPix(gry, corners, (11, 11), (-1, -1), criteria)
        
        objpoints.append(objp)
        imgpoints.append(corners2)

        cv.drawChessboardCorners(frame, pattern_size, corners2, ret)
        filename = os.path.join(out_dir, f"frame_{saved_frames:02d}.jpg")
        cv.imwrite(filename, frame)
        print(f"[INFO] Saved frame {saved_frames+1}/{total_needed}")
        saved_frames += 1

    
    # Show the current webcam frame
    msg = f"Saved: {saved_frames}/{total_needed} - q to exit"
    cv.putText(frame, msg, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    frame = cv.flip(frame, 1)
    cv.imshow("Calibration", frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

# --- Run camera calibration ---
print("[INFO] Running camera calibration...")

image_size = gry.shape[::-1]  # (width, height) of last processed frame

ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)

print(f"[INFO] Calibration successful: RMS error = {ret:.4f}")
print(f"[INFO] Camera matrix (K):\n{K}")
print(f"[INFO] Distortion coefficients:\n{dist}")

# Save calibration result to disk
np.savez(f"{out_dir}/intrinsics.npz", K=K, dist=dist)

print("[INFO] Calibration saved to 'intrinsics.npz'")

