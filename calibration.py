import numpy as np
import cv2
import time

# 定义相机内参和畸变系数
camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

# 生成棋盘格的三维坐标
objp = np.zeros((6*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

# 存储棋盘格角点的世界坐标和图像坐标
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane.

# 打开摄像头
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# 设置检测间隔为0.1秒
detection_interval = 0.1
last_detection_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # 将图像转换为灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 当当前时间距离上次检测超过0.1秒时进行检测
    if time.time() - last_detection_time >= detection_interval:
        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

        # 如果找到角点，则添加到objpoints和imgpoints中
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # 在图像上绘制角点
            cv2.drawChessboardCorners(frame, (8, 6), corners, ret)

            # 更新上次检测时间
            last_detection_time = time.time()

    # 显示实时图像
    cv2.imshow('Calibration', frame)

    # 按Esc键退出循环
    if cv2.waitKey(1) == 27:
        break

# 关闭摄像头
cap.release()
cv2.destroyAllWindows()

print(len(objpoints))

# 进行相机内参和畸变系数的标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 保存相机内参和畸变系数到文件
np.savez('camera_params.npz', mtx=mtx, dist=dist)

print("Calibration parameters saved to 'calibration_params.npz'")

