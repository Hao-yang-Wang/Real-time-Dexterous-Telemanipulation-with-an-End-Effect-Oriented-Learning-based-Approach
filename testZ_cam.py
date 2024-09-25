import torch
from rl_modules.models import actor
from arguments import get_args
import gym
import numpy as np
import transformations as tf
import cv2
import matplotlib.pyplot as plt 

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs



if __name__ == '__main__':

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
    
    args = get_args()
    
    # load the model param
    model_path = args.save_dir + args.env_name + '/model.pt'
    o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    
    # create the environment
    env = gym.make(args.env_name)
    
    # get the env param
    observation = env.reset()
    
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
                  
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    
    for i in range(5):

        observation = env.reset()

        env.initial_qpos = np.array([1.01570427,0.87487394,0.17090474,0.80575339,0.06654758,0.55336469,0.2003008])

        obs = observation['observation']
        
        fig, ax = plt.subplots()
        target_angles = []
        actual_angles = []
        
        pitch = 0
        yaw = 0 
        roll = 0
        
        for t in range(300): # env._max_episode_steps
        
            # 读取视频流
            ret, frame = cap.read()

            # 检测ARUCO码
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

            if ids is not None:
                # 估计位姿
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

                for i in range(len(rvec)):
                    cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec[i], tvec[i], 0.1)
                    
                    # Convert rotation vector to rotation matrix
                    R, _ = cv2.Rodrigues(rvec[i])

                    # Convert rotation matrix to Euler angles (pitch, yaw, roll)
                    pitch, yaw, roll = cv2.RQDecomp3x3(R)[0]
                    roll = -roll
                    # Print or use the Euler angles as needed
                    # print("Euler Angles (Pitch, Yaw, Roll):", pitch, yaw, roll)

                    
                # 在图像上绘制检测到的ARUCO码
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # 显示结果
            cv2.imshow('ARUCO Pose Estimation', frame)
        
            env.render()
            
            quaternion = tf.quaternion_from_euler(0 ,0, roll/180 * np.pi)
            pos = np.array([1.01570427,0.87487394,0.17090474])
            g = np.append(pos, quaternion) 
            
            #print(yaw)
            env.render_callback(g)
            
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            action[0] = action[0]

            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']
            acg = observation_new['achieved_goal']
            target_angles.append(roll/180 * np.pi)
            ox, oy ,oz = tf.euler_from_quaternion(acg[-4:])
            actual_angles.append(oz)
            
        #print('the episode is: {}, is success: {}'.format(i, info['is_success']))
        mse = np.mean((np.array(actual_angles) - np.array(target_angles)) ** 2)

        plt.plot(target_angles, label='Target')
        plt.plot(actual_angles, label='Actual ' + 'MSE:' + '%.3f'%mse)
        
        plt.title("Target tracking test")
        plt.xlabel("Timestep")
        plt.ylabel("Angle")
        
        plt.legend()
        plt.show()