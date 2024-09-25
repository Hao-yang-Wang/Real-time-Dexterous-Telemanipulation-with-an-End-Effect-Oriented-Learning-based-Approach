# Real-time Dexterous Telemanipulation with an End-Effect-Oriented  Learning-based Approach

This project implements a Reinforcement Learning (RL)-based robotic teleoperation system. The system uses ARUCO marker detection and built-in Inertial Measurement Unit (IMU) for pose estimation and tracks the target object’s orientation in real-time. The code integrates a Deep Deterministic Policy Gradient (DDPG) model for controlling the robot's movements in a simulated environment using OpenAI Gym.

## Requirements

### Python Libraries
To run this project, you need the following Python libraries:

- `torch` (PyTorch) – For deep learning and loading the actor model.
- `gym` – For the simulation environment.
- `numpy` – For numerical operations.
- `opencv-python` – For ARUCO marker detection and video processing.
- `cv2.aruco` – For ARUCO marker generation and detection.
- `matplotlib` – For plotting results and comparing target vs actual angles.
- `transforms3d` or `scipy` – For transformations (quaternions, Euler angles).
- `transformations` (can be found as `transforms3d` or via other packages).
- Any additional dependencies required by your custom modules like `rl_modules.models` and `arguments`.

Install the required packages using pip:

```bash
pip install torch gym numpy opencv-python matplotlib transforms3d
```

Or using `requirements.txt`, run:

```bash
pip install -r requirements.txt
```

### External Files
- `camera_params.npz`: This file contains pre-calibrated camera parameters (mtx for the camera matrix and dist for distortion coefficients).
- `model.pt`: This is the trained actor model used by the RL system.
Ensure these files are in the correct location as referenced in the code.


## Usage

- `Camera Calibration`: The code loads camera parameters from the camera_params.npz file. If you don't have pre-calibrated parameters, you may need to calibrate your camera and save them in this format use this script:

```bash
python calibration.py
```

- `Running the Code`: Once the dependencies are installed and the required files are in place, run the script:

```bash
python testZ_cam.py
```

- `Real-Time Pose Estimation`: The system captures frames from a connected camera (cv2.VideoCapture(1)) and detects ARUCO markers in real time. If ARUCO markers are found, their pose (rotation and translation) is estimated.

- `Environment Interaction`: The robot's control policy interacts with an OpenAI Gym environment, receiving observations and computing actions based on the target object’s current orientation (estimated from the ARUCO marker). The actor model calculates actions which are then applied to the environment.

- `Plotting`: After each episode, the code plots the target angles (roll angle in this case) vs. the actual angles achieved by the robot, along with the Mean Squared Error (MSE) of the tracking.
