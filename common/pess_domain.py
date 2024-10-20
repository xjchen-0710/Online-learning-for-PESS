# coding=utf-8
"""
Author:DYK
Email:y.d@pku.edu.cn

date:27/6/2022 下午8:55
desc:
"""

import numpy as np
import gym


class PESSFlattenedActionWrapper(gym.ActionWrapper):
    """
    Changes the format of the parameterised action space to conform to that of Goal-v0 and Platform-v0
    """
    def __init__(self, env, is_evaluate, episode_number):
        super(PESSFlattenedActionWrapper, self).__init__(env)
        old_as = env.action_space
        env.is_evaluate = is_evaluate
        env.episode_number = episode_number
        num_actions = old_as.spaces[0].n
        self.action_space = gym.spaces.Tuple((
            old_as.spaces[0],  # actions
            *(gym.spaces.Box(old_as.spaces[1].spaces[i].low, old_as.spaces[1].spaces[i].high, dtype=np.float32)
              for i in range(0, num_actions))
        ))

    def action(self, action):
        return action

