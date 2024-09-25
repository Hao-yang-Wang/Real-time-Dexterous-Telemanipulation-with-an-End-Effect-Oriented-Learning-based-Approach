import numpy as np
import cv2
import time

# Define camera intrinsic parameters and distortion coefficients
camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

# Generate 3D coordinates of the chessboard pattern
objp = np.zeros((6*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

# Store the world coordinates and image coordinates of the chessboard corners
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Open the camera
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Set detection interval to 0.1 seconds
detection_interval = 0.1
last_detection_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform detection if the current time exceeds the last detection time by more than 0.1 seconds
    if time.time() - last_detection_time >= detection_interval:
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

        # If corners are found, add them to objpoints and imgpoints
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw the chessboard corners on the image
            cv2.drawChessboardCorners(frame, (8, 6), corners, ret)

            # Update the last detection time
            last_detection_time = time.time()

    # Display the real-time image
    cv2.imshow('Calibration', frame)

    # Press Esc key to exit the loop
    if cv2.waitKey(1) == 27:
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()

print(len(objpoints))

# Perform camera calibration to obtain intrinsic parameters and distortion coefficients
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the camera intrinsic parameters and distortion coefficients to a file
np.savez('camera_params.npz', mtx=mtx, dist=dist)

print("Calibration parameters saved to 'calibration_params.npz'")


