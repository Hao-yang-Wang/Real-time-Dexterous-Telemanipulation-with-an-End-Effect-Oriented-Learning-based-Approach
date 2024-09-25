# Real-time Dexterous Telemanipulation with an End-Effect-Oriented  Learning-based Approach

This project implements a Reinforcement Learning (RL)-based robotic teleoperation system. The system uses ARUCO marker detection and built-in Inertial Measurement Unit (IMU) for pose estimation and tracks the target object’s orientation in real-time. The code integrates a deep deterministic policy gradient (DDPG) model for controlling the robot's movements in a simulated environment using OpenAI Gym.

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
