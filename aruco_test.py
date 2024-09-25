import cv2
import numpy as np

loaded_data = np.load('camera_params.npz')
loaded_camera_matrix = loaded_data['mtx']
loaded_dist_coeffs = loaded_data['dist']

# 设置相机参数
camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

# 创建ARUCO字典
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# 创建ARUCO检测器
parameters = cv2.aruco.DetectorParameters_create()

# 打开摄像头
cap = cv2.VideoCapture(1)

while True:
    # 读取视频流
    ret, frame = cap.read()

    # 检测ARUCO码
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        # 估计位姿
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

        # 在图像上绘制坐标轴
        for i in range(len(rvec)):
            cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec[i], tvec[i], 0.1)

    # 在图像上绘制检测到的ARUCO码
    frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # 显示结果
    cv2.imshow('ARUCO Pose Estimation', frame)

    # 按下ESC键退出循环
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()




"""
cv2.putText(frame, 'Rotation Angle: {:.2f}'.format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
"""







