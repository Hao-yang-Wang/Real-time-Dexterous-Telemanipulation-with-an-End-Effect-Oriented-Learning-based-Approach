"""Wrapper for Global Observation"""
import gym
import numpy as np
from gym.envs.robotics import rotations

class GlobalObservation(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.distance_threshold2 = 0.01
        self.rotation_threshold2 = 0.03
    
    def render_callback(self, goal):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.
        goal = goal.copy()
        
        assert goal.shape == (7,)
        if self.target_position == 'ignore':
            # Move the object to the side since we do not care about it's position.
            goal[0] += 0.15
            
        self.sim.data.set_joint_qpos('target:joint', goal)
        self.sim.data.set_joint_qvel('target:joint', np.zeros(6))

        if 'object_hidden' in self.sim.model.geom_names:
            hidden_id = self.sim.model.geom_name2id('object_hidden')
            self.sim.model.geom_rgba[hidden_id, 3] = 1.
        self.sim.forward()
    
    def _goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        assert goal_a.shape[-1] == 7

        d_pos = np.zeros_like(goal_a[..., 0])
        d_rot = np.zeros_like(goal_b[..., 0])
        if self.target_position != 'ignore':
            delta_pos = goal_a[..., :3] - goal_b[..., :3]
            d_pos = np.linalg.norm(delta_pos, axis=-1)

        if self.target_rotation != 'ignore':
            quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]

            if self.ignore_z_target_rotation:
                # Special case: We want to ignore the Z component of the rotation.
                # This code here assumes Euler angles with xyz convention. We first transform
                # to euler, then set the Z component to be equal between the two, and finally
                # transform back into quaternions.
                euler_a = rotations.quat2euler(quat_a)
                euler_b = rotations.quat2euler(quat_b)
                euler_a[2] = euler_b[2]
                quat_a = rotations.euler2quat(euler_a)

            # Subtract quaternions and extract angle between them.
            quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
            angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
            d_rot = angle_diff
        assert d_pos.shape == d_rot.shape
        return d_pos, d_rot
    
    def _is_success2(self, achieved_goal, desired_goal):
        d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
        achieved_pos = (d_pos < self.distance_threshold2).astype(np.float32)
        achieved_rot = (d_rot < self.rotation_threshold2).astype(np.float32)
        achieved_both = achieved_pos * achieved_rot
        return achieved_both, d_pos, d_rot
    
    def compute_reward(self, achieved_goal, goal, info):
        success, d_pos, d_rot = self._is_success2(achieved_goal, goal)
        success = success.astype(np.float32)
        return -(d_pos + d_rot) + (success - 1.)
          
    def reset(self):
        ob = self.env.reset()
        return ob

    def step(self, action):
        ob, _, done, info = self.env.step(action)   
        reward = self.compute_reward(ob['achieved_goal'], ob['desired_goal'], None)
        return ob, reward, done, info

