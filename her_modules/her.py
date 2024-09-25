import numpy as np

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        # 初始化HER采样器
        # replay_strategy: 回放策略，可以是'future'（未来经验回放）或其他值（默认）
        # replay_k: K值，影响未来经验回放的概率
        # reward_func: 用于计算奖励的函数，默认为None
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        self.action = None
        
        if self.replay_strategy == 'future':
            # 如果采用未来经验回放策略，计算未来经验回放的概率
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        # 从经验池中采样HER样本
        # episode_batch: 一个字典，包含了一个完整的轨迹的所有信息（状态、动作、奖励等）
        # batch_size_in_transitions: 采样的样本数

        T = episode_batch['actions'].shape[1]  # 时间步数
        rollout_batch_size = episode_batch['actions'].shape[0]  # 轨迹数量
        batch_size = batch_size_in_transitions

        # 随机选择要使用的轨迹和时间步
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        # HER索引
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # 用达到的目标替换目标
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # 获取重新计算奖励的参数
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)

        # 重新整理样本数据的形状
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions
