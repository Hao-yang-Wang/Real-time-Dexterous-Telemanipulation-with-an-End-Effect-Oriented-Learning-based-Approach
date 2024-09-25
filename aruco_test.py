import cv2
import numpy as np

loaded_data = np.load('camera_params.npz')
loaded_camera_matrix = loaded_data['mtx']
loaded_dist_coeffs = loaded_data['dist']

camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

parameters = cv2.aruco.DetectorParameters_create()

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    if ids is not None:
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
        for i in range(len(rvec)):
            cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec[i], tvec[i], 0.1)
    frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow('ARUCO Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()







