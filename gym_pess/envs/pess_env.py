# coding=utf-8
"""
Author:DYK
Email:y.d@pku.edu.cn

date:24/6/2022 下午9:51
desc:
"""
import numpy as np
import gym
import os
from os import path
from gym import error, spaces, utils
from gym.utils import seeding
import sys
import pickle
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from load_data import *
from node_data_processing import *
from deg_nn.deg_api import MLPregression, zscores, model

# actions
CHARGE = "charge"
DISCHARGE = "discharge"
HOLD = "hold"

ACTION_LOOKUP = {
    0: CHARGE,
    1: DISCHARGE,
    2: HOLD,
}



class PESSEnv(gym.Env):

    def __init__(self):
        """ Setup environment """

        self.ENERGY_MIN = 0.0
        self.ENERGY_MAX = 2.7
        self.MAX_CHARGE_PWR = 2.7
        self.MIN_CHARGE_PWR = 0.0
        self.EFF = 0.95
        self.STARTING_ENERGY = 0.0
        self.data_path = None
        self.NODES = 1
        self.center_node_name = None
        self.sample_number = '36'
        self.radius = 10.0
        self.TRA_COST = 20
        self.SCALE = 0.25
        self.day_set = [81]

        # self.DAY = self.day_set[0]
        self.DAY = self.day_set[np.random.randint(0, (len(self.day_set)))]
        # self.STAY_MAX = 24 / self.SCALE * 2

        self.PARAMETERS_MIN = np.array([0, 0, 0])
        self.PARAMETERS_MAX = np.array([
            2.7,  # charge
            2.7,  # discharge
            0  # hold
        ])

        self.pess = PESS()


        # self.MAX_TIME_STEP = len(self.day_set) * 24 / self.SCALE
        self.MAX_TIME_STEP = 1 * 24 / self.SCALE
        self.time_step = 0

        self.np_random = None
        self.seed()

        self.states = []
        self.state_begin = np.zeros(int(self.MAX_TIME_STEP))
        self.pess.history_cost = 0
        self.render_states = []

        cwd = os.getcwd()
        self.data_path = cwd + '/data/'

        self.is_evaluate = False
        self.episode_number = 0

        # [location_data, node_id_set, duration_data, distance_data, LMP_data, LMP_data_RTM, price_data_SR,
        #  price_data_NR, node_data_clean] = load_data(self.data_path)
        [LMP_data, LMP_data_RTM, node_data_clean] = load_data(self.data_path)

        # 加载老化模型，以及标准化文件
        # self.zscores = pickle.load(open(cwd + '/deg_nn/' + 'zscores.pkl', 'rb'))
        self.zscores = zscores

        # self.deg_model = torch.load(cwd + '/deg_nn/' + 'model.pth')

        self.deg_model = model
        self.deg_model.eval()

        self.LMP_data = LMP_data
        self.LMP_data_RTM = LMP_data_RTM

        self.temp_all_nodes = [
            [10.2, 9.3, 9.7, 9.1, 8.6, 8.4, 7.6, 8.2, 10.2, 11.9, 13.7, 15.1, 15.2, 16.2, 15.8, 15.5, 15.1, 14.4, 13.4,
             13, 12.7, 12.1, 12.1, 11.4],
            [8.5, 7.5, 8.6, 8.1, 7.5, 7.2, 5.5, 7.3, 9.4, 11.9, 13.8, 14.7, 15.5, 16.3, 15.7, 15.3, 14.7, 13.8, 12.7,
             11.9, 11.5, 10.9, 10.8, 9.8],
            [10.9, 10.1, 10.1, 9.6, 8.9, 8.9, 8.6, 8.4, 10.4, 12.1, 13.6, 15.3, 15.2, 16.3, 15.9, 15.6, 15.3, 14.6,
             13.7, 13.4, 13.1, 12.7, 12.6, 11.8],
            [10.8, 9.9, 10, 9.5, 8.9, 8.9, 8.4, 8.4, 10.4, 12, 13.7, 15.3, 15.1, 16.3, 15.9, 15.6, 15.3, 14.5, 13.7,
             13.3, 13, 12.6, 12.5, 11.7],
            [10.9, 10.1, 10.1, 9.7, 9, 9, 8.6, 8.5, 10.5, 12.1, 13.7, 15.3, 15.1, 16.4, 15.9, 15.6, 15.3, 14.6, 13.7,
             13.5, 13.1, 12.7, 12.6, 11.9],
            [11.1, 10.2, 10.2, 9.8, 9.1, 9.1, 8.8, 8.6, 10.6, 12.1, 13.7, 15.4, 15.1, 16.3, 15.9, 15.7, 15.4, 14.7,
             13.8, 13.5, 13.2, 12.8, 12.7, 12],
            [10.6, 9.8, 9.8, 9.5, 8.7, 8.7, 8.3, 8.3, 10.3, 12.2, 13.7, 15.3, 15.3, 16.5, 15.9, 15.7, 15.3, 14.5, 13.6,
             13.2, 12.9, 12.5, 12.2, 11.4],
            [11.4, 10.4, 10.4, 10.2, 9.5, 9.5, 9, 8.9, 10.8, 12.1, 13.7, 15.5, 15, 16.5, 16, 15.9, 15.5, 14.8, 13.8,
             13.7, 13.3, 12.8, 12.7, 12.1],
            [11.7, 10.6, 10.6, 10.6, 9.9, 9.9, 9.3, 9.3, 11.1, 12.2, 13.8, 15.6, 15, 16.7, 16.1, 16.1, 15.6, 15, 13.8,
             13.8, 13.3, 12.8, 12.8, 12.2],
            [9.4, 8.7, 8.9, 8.9, 8.1, 8, 7.3, 8, 10.3, 12.7, 14.1, 15.4, 16, 17.1, 16.1, 16, 15.3, 14.5, 13.3, 12.7,
             12.2, 11.4, 11, 10.2],
            [9.9, 9.2, 9.2, 9.2, 8.3, 8.3, 7.9, 8.2, 10.5, 12.7, 14, 15.5, 15.9, 17, 16.1, 16, 15.4, 14.6, 13.5, 13,
             12.6, 11.8, 11.4, 10.6],
            [8.6, 7.8, 8.4, 8.2, 7.5, 7.3, 6.3, 7.6, 9.9, 12.6, 14.1, 15.2, 16.1, 17, 16, 15.8, 15.1, 14.2, 13, 12.2,
             11.7, 10.9, 10.5, 9.5],
            [10.1, 9.3, 9.3, 9.3, 8.5, 8.5, 8, 8.3, 10.5, 12.6, 14, 15.5, 15.8, 17, 16, 16, 15.4, 14.6, 13.5, 13.1,
             12.6, 12, 11.6, 10.8],
            [10.5, 9.7, 9.7, 9.6, 8.8, 8.8, 8.3, 8.5, 10.6, 12.5, 13.9, 15.5, 15.5, 16.9, 16, 16, 15.4, 14.6, 13.5,
             13.2, 12.8, 12.2, 11.9, 11.1],
            [9.9, 9.2, 9.2, 9.2, 8.3, 8.3, 7.8, 8.1, 10.3, 12.7, 14, 15.4, 15.8, 17, 16, 15.9, 15.3, 14.4, 13.3, 12.9,
             12.4, 12, 11.4, 10.5],
            [9.3, 8.6, 8.6, 8.7, 7.8, 7.8, 7.3, 7.8, 10.1, 12.9, 14.1, 15.4, 16.2, 17.3, 16, 16, 15.3, 14.3, 13.2, 12.6,
             12.1, 11.5, 10.8, 9.8],
            [9.1, 8.4, 8.4, 8.7, 7.8, 7.8, 7.4, 8.1, 10.7, 13.2, 14.5, 15.7, 16.5, 17.5, 16.3, 16.3, 15.5, 14.7, 13.4,
             12.8, 12.2, 10.9, 10.5, 9.8],
            [9.1, 8.4, 8.4, 8.7, 7.8, 7.8, 7.5, 8.1, 10.8, 13.3, 14.5, 15.8, 16.5, 17.5, 16.3, 16.3, 15.6, 14.8, 13.4,
             12.8, 12.2, 10.9, 10.5, 9.8],
            [9.1, 8.4, 8.4, 8.7, 7.8, 7.8, 7.5, 8.1, 10.8, 13.3, 14.5, 15.8, 16.5, 17.5, 16.3, 16.3, 15.6, 14.8, 13.4,
             12.8, 12.3, 10.9, 10.5, 9.8],
            [9.1, 8.4, 8.4, 8.7, 7.8, 7.8, 7.5, 8.1, 10.8, 13.3, 14.5, 15.8, 16.5, 17.5, 16.3, 16.3, 15.6, 14.8, 13.5,
             12.8, 12.3, 10.9, 10.5, 9.8],
            [9.2, 8.5, 8.5, 8.7, 7.9, 7.9, 7.5, 8.2, 10.9, 13.3, 14.5, 15.8, 16.5, 17.5, 16.4, 16.4, 15.6, 14.8, 13.5,
             12.8, 12.3, 10.9, 10.5, 9.8],
            [8.7, 8.1, 8.1, 8.3, 7.4, 7.4, 7.1, 7.7, 10.5, 13.3, 14.5, 15.6, 16.6, 17.6, 16.2, 16.2, 15.4, 14.5, 13.3,
             12.5, 12, 10.8, 10.2, 9.4],
            [8.8, 8.3, 8.3, 8.3, 7.2, 7.2, 6.7, 7.2, 9.4, 12.9, 13.9, 15.1, 16.2, 17.2, 15.7, 15.7, 15.1, 13.9, 12.9,
             12.2, 11.7, 11.6, 10.6, 9.4],
            [8.9, 8.3, 8.3, 8.5, 7.5, 7.5, 7.2, 7.8, 10.4, 13.2, 14.4, 15.6, 16.5, 17.5, 16.2, 16.2, 15.4, 14.5, 13.3,
             12.6, 12, 11, 10.4, 9.5],
            [8.7, 8.2, 8.2, 8.3, 7.3, 7.3, 6.9, 7.5, 10.1, 13.2, 14.3, 15.4, 16.5, 17.4, 16, 16, 15.3, 14.3, 13.1, 12.4,
             11.9, 11.1, 10.3, 9.3],
            [8.6, 8.1, 8.1, 8.2, 7.3, 7.3, 6.9, 7.6, 10.2, 13.2, 14.3, 15.5, 16.6, 17.5, 16.1, 16.1, 15.3, 14.4, 13.1,
             12.4, 11.9, 11, 10.2, 9.3],
            [7.4, 6.6, 6.3, 6.4, 5.8, 5.9, 5.4, 6.2, 9.8, 12.9, 14.2, 15.3, 16.4, 17.4, 16.4, 16.3, 15.5, 14.3, 12.4,
             11.4, 10.8, 8.9, 8.4, 7.9],
            [8.7, 8.1, 8.1, 8.4, 7.6, 7.6, 7.3, 8, 11, 13.5, 14.7, 15.9, 16.8, 17.7, 16.5, 16.5, 15.6, 14.9, 13.5, 12.7,
             12.2, 10.5, 10.1, 9.5],
            [6.6, 5.7, 5.1, 5.1, 4.8, 4.9, 4.1, 4.9, 8.9, 12.3, 13.6, 14.8, 15.9, 17, 16.4, 16.2, 15.4, 13.8, 11.7,
             10.5, 9.8, 8, 7.3, 6.9],
            [8.1, 7.5, 7.5, 8, 7.3, 7.3, 7.2, 8.2, 11.7, 14, 15.3, 16.4, 17.5, 18.1, 16.9, 16.9, 16, 15.4, 13.7, 12.8,
             12.1, 9.4, 9.3, 9],
            [8.3, 7.6, 7.6, 8.1, 7.4, 7.4, 7.2, 8.1, 11.5, 13.9, 15.1, 16.2, 17.3, 18, 16.8, 16.8, 15.9, 15.2, 13.6,
             12.8, 12.1, 9.7, 9.5, 9.2]]

        self.temp_all_nodes_rescale = np.zeros((31, 96))

        scale = 0.25

        for n in range(31):
            for t in range(96):
                self.temp_all_nodes_rescale[n][t] = self.temp_all_nodes[n][floor(t * scale)]






        c_index = int(self.sample_number)
        self.center_node_name = node_data_clean['node_name'][c_index]
        node_data = node_data_clean
        self.node_set = load_node_set_circle(self.center_node_name, self.radius, node_data)
        self.node_num = len(self.node_set)

        print("Nodes number: ", self.node_num)

        self.OBSERVATION_MIN = np.array([0, 0, 0])
        self.OBSERBATION_MAX = np.array([
            2.7,  # energy
            np.Inf,  # system time
            self.node_num  # node numbers
        ])

        self.pess.current_node = 0

        [self.nodes_duration, _] = travel_data_cal(self.node_set, node_data)

        self.p_da_rescale = self.get_day_price(self.node_set, self.DAY, self.day_set, LMP_data)

        self.p_rt_rescale = self.get_rt_price(self.node_set, self.DAY, self.day_set, LMP_data_RTM)
        # self.p_rt_rescale = self.p_da_rescale

        # num_actions = node_num
        soc_num = len(ACTION_LOOKUP)
        num_actions = soc_num * self.node_num
        # action_space:(3,n,pwr)
        self.action_space = spaces.Tuple((
            spaces.Discrete(num_actions),
            spaces.Tuple(
                tuple(spaces.Box(low=np.array([self.PARAMETERS_MIN[i]]),
                                 high=np.array([self.PARAMETERS_MAX[i]]), dtype=np.float32)
                      for j in range(self.node_num) for i in range(soc_num))
            )
        ))

        self.history_params = np.zeros(int(self.MAX_TIME_STEP))

        self.observation_space = spaces.Tuple((
            spaces.Box(low=-np.inf, high=np.inf, shape=self.get_state(self.time_step, 0, 0).shape, dtype=np.float32),
            spaces.Discrete(1000),  # steps (200 limit is an estimate)
        ))

    # self.observation_space = spaces.Tuple((
    # 		tuple(spaces.Box(low=np.array([self.OBSERVATION_MIN[i]]),
    # 						 high=np.array([self.OBSERBATION_MAX[i]]), dtype=np.float32)
    # 	for i in range(num_actions)),
    # 	spaces.Discrete(1000)
    # ))

    def set_episode_value(self, episode):
        self.episode_number = episode
        # print("episode number: ", self.episode_number)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        # self._update_seeds()
        return [seed]

    # def _update_seeds(self):

    def step(self, action):
        action_index = action[0] % 3
        next_node = action[0] // 3
        act = ACTION_LOOKUP[action_index]

        episode_number = self.episode_number

        # if act == DISCHARGE:
        # 	print("DISCHARGE")

        # # 随机获取七天内的电价
        # self.DAY = self.day_set[np.random.randint(0, len(self.day_set))]
        # # print("DAY: ", self.DAY)
        # self.p_da_rescale = self.get_day_price(self.node_set, self.DAY, self.day_set, self.LMP_data)
        # # print(self.p_da_rescale)
        # self.p_rt_rescale = self.get_rt_price(self.node_set, self.DAY, self.day_set, self.LMP_data_RTM)

        print("capacity: ", self.pess.ENERGY_MAX, "SOH: ", self.pess.soh, "current energy: ", self.pess.energy_level,
              "current node: ", self.pess.current_node, "time step: ", self.time_step)

        param = action[1][next_node * 3 + action_index][0]
        param = np.clip(param, self.PARAMETERS_MIN[action_index], self.PARAMETERS_MAX[action_index])

        self.pess.step_reward = 0.

        duration_time = np.ceil(self.nodes_duration[self.pess.current_node][next_node] / (60 * self.SCALE)).astype(int)

        state_before = self.get_state(self.time_step, param, duration_time)
        obs_before = (state_before, self.time_step)

        # end_episode = False

        e1 = self.pess.energy_level

        # duration_time = 0
        end_episode = self._update(act, next_node, param, duration_time, self.is_evaluate, episode_number)

        e2 = self.pess.energy_level
        # self.history_params[self.time_step] = param
        if not end_episode:
            state = self.get_state(self.time_step, param, duration_time)
            obs = (state, self.time_step)
            return obs, self.pess.step_reward, end_episode, {}
        else:
            return obs_before, self.pess.step_reward, end_episode, {}

    def _update(self, act, next_node, param, duration_time, is_evaluate, episode_number):

        # self.render_states.append(self.states[-1])
        self._perform_action(act, next_node, param, duration_time, is_evaluate, episode_number)
        return self._terminal_check()

    def reset(self):
        self.pess.reset()
        self.time_step = 0
        self.pess.current_node = 0
        self.state_begin = np.zeros(int(self.MAX_TIME_STEP))
        self.states = []
        self.history_params = np.zeros(int(self.MAX_TIME_STEP))
        return self.get_state(self.time_step, 0, 0), 0

    def _perform_action(self, act, next_node, parameters, duration_time, is_evaluate, episode_number):
        """ Applies for selected action for the given agent. """

        if (self.time_step + duration_time + 1 > self.MAX_TIME_STEP):
            self.time_step += (1 + duration_time)
            return self._terminal_check()

        else:

            if act == CHARGE:
                self.pess.charge(next_node, parameters, duration_time, self.p_da_rescale, self.p_rt_rescale, self.SCALE,
                                 self.time_step, is_evaluate, self.zscores, self.deg_model, episode_number, self.temp_all_nodes_rescale)
            elif act == DISCHARGE:
                self.pess.discharge(next_node, parameters, duration_time, self.p_da_rescale, self.p_rt_rescale,
                                    self.SCALE, self.time_step, is_evaluate, self.zscores, self.deg_model, episode_number)
            elif act == HOLD:
                self.pess.hold(next_node, duration_time, self.SCALE)
            else:
                raise error.InvalidAction("Action not recognised: ", act)

            self.time_step += (1 + duration_time)

            return self._terminal_check()

    def _terminal_check(self):
        """ Determines whether the episode is ended, and the reward. """
        end_episode = False
        if self.time_step >= self.MAX_TIME_STEP - 1:
            end_episode = True

        return end_episode

    def get_state(self, time_step, param, duration_time):
        """ Returns the representation of the current state.
		:param duration_time:
		:param time_step:
		"""
        # state = np.array((
        # 	self.pess.energy_level,
        # 	self.p_da_rescale[self.pess.current_node][self.time_step],
        # 	self.pess.current_node,
        # ), dtype=np.float32)
        #
        # return state

        ###################################################################################################
        # predict_steps = 40
        #
        # predictions = []
        # if (time_step < int(self.MAX_TIME_STEP - predict_steps)):
        # 	predictions = [self.p_da_rescale[self.pess.current_node][self.time_step + i] for i in range(predict_steps)]
        # elif (time_step >= int(self.MAX_TIME_STEP - predict_steps)):
        # 	predictions = [self.p_da_rescale[self.pess.current_node][self.time_step + i] for i in
        # 				   range(int(self.MAX_TIME_STEP - time_step))]
        # 	padding_zeros = [0 for i in range(int(predict_steps - (self.MAX_TIME_STEP - time_step)))]
        # 	predictions = np.concatenate((predictions, padding_zeros))
        #
        # # predictions = [self.p_da_rescale[self.pess.current_node][i] for i in
        # # 				   range(int(self.MAX_TIME_STEP))]
        #
        # state = np.concatenate((np.array((self.pess.energy_level,), dtype=np.float32),
        # 						np.array((predictions), dtype=np.float32),
        # 						np.array((self.pess.current_node,), dtype=np.float32),
        # 						np.array((param,), dtype=np.float32)))

        ############################## history price #######################################

        # self.state_begin[time_step] = self.p_da_rescale[self.pess.current_node][self.time_step]
        # state = np.concatenate((np.array((self.pess.energy_level,), dtype=np.float32),
        # 						np.array((self.state_begin), dtype=np.float32),
        # 						np.array((self.pess.current_node,), dtype=np.float32),
        # 						np.array((param,), dtype=np.float32)))

        ################################# history cost ############################################

        current_price_all_nodes = [self.p_da_rescale[i][self.time_step] for i in range(len(self.node_set))]

        # 温度
        current_temp_all_nodes = [self.temp_all_nodes_rescale[i][self.time_step] for i in range(len(self.node_set))]

        current_rt_price_all_nodes = [self.p_rt_rescale[i][self.time_step] for i in range(len(self.node_set))]
        # before_rt_price_all_nodes = []
        before_steps = 16
        if (self.time_step >= int(before_steps)):
            before_rt_price_all_nodes = [self.p_rt_rescale[self.pess.current_node][self.time_step - i] for i in
                                         range(before_steps, 0, -1)]
        elif (self.time_step < int(before_steps)):
            before_rt_price_all_nodes = [self.p_rt_rescale[self.pess.current_node][self.time_step - i] for i in
                                         range(int(self.time_step), 0, -1)]
            padding_zeros = [0 for i in range(int(before_steps - self.time_step))]
            before_rt_price_all_nodes = np.concatenate((padding_zeros, before_rt_price_all_nodes))

        # 前16步日前电价，用于：训练和对齐实时电价格式
        if (self.time_step >= int(before_steps)):
            before_price_all_nodes = [self.p_da_rescale[self.pess.current_node][self.time_step - i] for i in
                                      range(before_steps, 0, -1)]
        elif (self.time_step < int(before_steps)):
            before_price_all_nodes = [self.p_da_rescale[self.pess.current_node][self.time_step - i] for i in
                                      range(int(self.time_step), 0, -1)]
            padding_zeros = [0 for i in range(int(before_steps - self.time_step))]
            before_price_all_nodes = np.concatenate((padding_zeros, before_price_all_nodes))

        # predictions = [self.p_da_rescale[self.pess.current_node][i] for i in
        # 				   range(int(self.MAX_TIME_STEP))]

        if (self.time_step + duration_time < self.MAX_TIME_STEP):
            next_price_all_nodes = [self.p_da_rescale[i][self.time_step + duration_time] for i in
                                    range(len(self.node_set))]
        else:
            next_price_all_nodes = current_price_all_nodes

        if (self.time_step + duration_time < self.MAX_TIME_STEP):
            next_rt_price_all_nodes = [self.p_rt_rescale[i][self.time_step + duration_time] for i in
                                       range(len(self.node_set))]
        else:
            next_rt_price_all_nodes = current_rt_price_all_nodes

        current_price = self.p_da_rescale[self.pess.current_node][self.time_step]

        # time_steps = [self.time_step] * 20

        if self.is_evaluate:
            state = np.concatenate((np.array((self.pess.energy_level,), dtype=np.float32),
                                    np.array((self.pess.history_cost,), dtype=np.float32),
                                    # np.array((current_price_all_nodes), dtype=np.float32),
                                    np.array((current_rt_price_all_nodes), dtype=np.float32),
                                    # np.array((next_price_all_nodes), dtype=np.float32),
                                    np.array((next_rt_price_all_nodes), dtype=np.float32),
                                    np.array((before_rt_price_all_nodes), dtype=np.float32),
                                    np.array((current_price,), dtype=np.float32),
                                    np.array((self.time_step,), dtype=np.float32),
                                    np.array((self.pess.current_node,), dtype=np.float32)
                                    ))
        else:
            state = np.concatenate((np.array((self.pess.energy_level,), dtype=np.float32),
                                    np.array((self.pess.history_cost,), dtype=np.float32),
                                    np.array((current_price_all_nodes), dtype=np.float32),
                                    # np.array((current_rt_price_all_nodes), dtype=np.float32),
                                    np.array((next_price_all_nodes), dtype=np.float32),
                                    # np.array((before_price_all_nodes), dtype=np.float32),
                                    # np.array((next_rt_price_all_nodes), dtype=np.float32),
                                    # np.array((before_rt_price_all_nodes), dtype=np.float32),
                                    np.array((current_price,), dtype=np.float32),
                                    np.array((self.time_step,), dtype=np.float32),
                                    np.array((self.pess.current_node,), dtype=np.float32)
                                    ))

        ############################### history power###############################################
        # state = np.concatenate((np.array((self.pess.energy_level,), dtype=np.float32),
        # 						np.array((self.history_params), dtype=np.float32),
        # 						np.array((self.pess.current_node,), dtype=np.float32)))

        return state

    def get_day_price(self, node_set, day, day_set, LMP_data):
        p_da = []  # 日前电价
        node_num = len(node_set)

        # for i in range(len(node_set)):
        # 	temp = []
        # 	for d in day_set:
        # 		temp += list(LMP_data[node_set[i]][d])
        # 	p_da.append(temp)
        #
        for i in range(len(node_set)):
            p_da.append(LMP_data[node_set[i]][day])

        # p_da = np.stack(p_da, axis=0)

        hour_num = len(p_da[0])

        horizon = int(hour_num / self.SCALE)

        p_da_rescale = np.zeros(shape=(len(node_set), horizon))
        for n in range(node_num):
            for t in range(horizon):
                p_da_rescale[n][t] = p_da[n][floor(t * self.SCALE)]

        return p_da_rescale

    def get_rt_price(self, node_set, day, day_set, LMP_data_RTM):
        p_rt = []  # 实时电价
        node_num = len(node_set)
        # for d in day_set:
        # temp = []
        for i in range(len(node_set)):
            # temp.append(list(LMP_data_RTM[node_set[i]][d]))
            p_rt.append(list(LMP_data_RTM[node_set[i]][day]))  # p_rt代表每个点每天96个时段的电价（间隔为15分钟）
        #             p_rt.append(LMP_data[node_set[i]][day])

        hour_num = len(p_rt[0])
        horizon = int(hour_num / self.SCALE)
        p_rt_rescale = np.zeros(shape=(len(node_set), horizon))

        for n in range(node_num):
            for t in range(horizon):
                horizon_rt = len(p_rt[n])
                if floor(t * self.SCALE * 4) >= horizon_rt:
                    p_rt_rescale[n][t] = p_rt[n][horizon_rt - 1]
                else:
                    p_rt_rescale[n][t] = p_rt[n][floor(t * self.SCALE * 4)]

        return p_rt_rescale

    def reset_episode_step(self):
        self.time_step = 0


class PESS():

    def __init__(self):
        self.ENERGY_MIN = 0.0
        self.ENERGY_MAX = 2.7
        self.MAX_CHARGE_PWR = 2.7
        self.EFF = 0.95
        self.DEG_COST = 50.0
        self.STARTING_ENERGY = 0.0
        self.TRA_COST = 20
        self.energy_level = 0.0
        self.current_node = None
        self.history_cost = 0.0
        self.soh = 1

        self.step_reward = 0.

    def reset(self):
        self.energy_level = 0.0
        self.history_cost = 0.0
        self.ENERGY_MAX = 2.7
        self.soh = 1

    # self.current_node = None

    # def deg_calculate(self, pwr, soc_start, soc_end, soh, temper):
    #     data_test = pd.DataFrame([[pwr, soc_start, soc_end, soc_end, 25]],
    #                              columns=['set_P', 'SOC_start', 'SOC_end', 'set_SOH', 'temper'])


    def charge(self, next_node, pwr, duration_time, p_da_rescale, p_rt_rescale, SCALE, time_step, is_evaluate, zscores, deg_model, episode_number, temp_all_nodes_rescale):

        tra_cost = self.TRA_COST * duration_time * SCALE

        max_price = p_da_rescale[next_node][0]
        max_price = min(max_price, p_da_rescale[next_node][time_step])

        charge_energy = pwr * SCALE
        pre_energy = self.energy_level
        soc_start = pre_energy / self.ENERGY_MAX
        self.energy_level += charge_energy

        # delta_energy = charge_energy
        if self.energy_level > self.ENERGY_MAX:
            penalty = 50
            delta_energy = self.ENERGY_MAX - pre_energy
            self.energy_level = self.ENERGY_MAX
        # print("charge out of range: ", charge_energy)
        else:
            penalty = 0
            delta_energy = charge_energy

        soc_end = self.energy_level / self.ENERGY_MAX

        if episode_number > 1500 and (soc_end - soc_start) > 0.05:
            # 老化量计算
            pwr_map = pwr * 200 / 2.7
            # 真实温度
            # data_test = pd.DataFrame([[pwr_map, soc_start, soc_end, self.soh, temp_all_nodes_rescale[self.current_node][time_step], 0]],
            #                          columns=['set_P', 'SOC_start', 'SOC_end', 'set_SOH', 'temper','delta_SOH'])

            # 常数温度
            data_test = pd.DataFrame(
                [[pwr_map, soc_start, soc_end, self.soh, 25, 0]],
                columns=['set_P', 'SOC_start', 'SOC_end', 'set_SOH', 'temper', 'delta_SOH'])

            df_train = zscores.transform(data_test)
            pd_temp = (
                pd.DataFrame(df_train, columns=['set_P', 'SOC_start', 'SOC_end', 'set_SOH', 'temper', 'delta_SOH']))
            temp_trans = torch.from_numpy(
                pd_temp[['set_P', 'SOC_start', 'SOC_end', 'set_SOH', 'temper']].values.astype(np.float32))

            # df_train = torch.from_numpy(df_train.astype(np.float32))
            # deg = deg_model(temp_trans).detach().numpy()[0]

            deg = deg_model(temp_trans)

            ans = pd.DataFrame(deg.detach().numpy())
            ans_pro = pd.concat([pd_temp[['set_P', 'SOC_start', 'SOC_end', 'set_SOH', 'temper']], ans], axis=1)
            ans_soh = pd.DataFrame(zscores.inverse_transform(ans_pro)).iloc[:, -1]  # 得出最终的delta_SOH
            deg = ans_soh.values[0]

            if deg > 0:
                print(1)

            # 根据老化量更新最大容量，以及SOH
            self.soh += deg
            self.ENERGY_MAX = (1 + deg) * self.ENERGY_MAX


        charge_cost = p_da_rescale[next_node][time_step] * delta_energy
        delta_price = max_price - p_da_rescale[next_node][time_step]
        self.current_node = next_node

        # use history cost
        # self.history_cost += p_da_rescale[next_node][time_step + duration_time] * delta_energy

        if is_evaluate:
            # use RT price to compute cost
            self.history_cost += p_rt_rescale[next_node][time_step + duration_time] * pwr * SCALE
        else:
            # use pwr to compute cost
            self.history_cost += p_da_rescale[next_node][time_step + duration_time] * pwr * SCALE

        # self.step_reward = -1 * charge_cost - tra_cost
        # self.step_reward = delta_price * delta_energy

        # use history cost
        self.step_reward = -self.DEG_COST * delta_energy - tra_cost

    def discharge(self, next_node, pwr, duration_time, p_da_rescale, p_rt_rescale, SCALE, time_step, is_evaluate, zscores, deg_model, episode_number):

        tra_cost = self.TRA_COST * duration_time * SCALE

        min_price = p_da_rescale[next_node][0]
        min_price = min(min_price, p_da_rescale[next_node][time_step])

        pre_energy = self.energy_level
        soc_start = pre_energy / self.ENERGY_MAX
        discharge_energy = pwr * SCALE
        self.energy_level -= discharge_energy

        # delta_energy = discharge_energy
        if self.energy_level < self.ENERGY_MIN:
            penalty = 50
            delta_energy = pre_energy
            self.energy_level = self.ENERGY_MIN
        # print("discharge out of range: ", discharge_energy)
        else:
            penalty = 0
            delta_energy = discharge_energy

        soc_end = self.energy_level / self.ENERGY_MAX

        discharge_revenue = p_da_rescale[next_node][time_step] * delta_energy
        self.current_node = next_node
        delta_price = p_da_rescale[next_node][time_step] - min_price
        # self.step_reward = discharge_revenue - tra_cost
        # self.step_reward = delta_price * delta_energy

        ######### 放电暂不考虑SOH衰减
        # if episode_number > 1500 and (soc_start - soc_end) > 0.05:
        #     # 老化量计算
        #     pwr_map = pwr * 350 / 2.7
        #     data_test = pd.DataFrame([[pwr_map, soc_start, soc_end, self.soh, 25, 0]],
        #                              columns=['set_P', 'SOC_start', 'SOC_end', 'set_SOH', 'temper', 'delta_SOH'])
        #     df_train = zscores.transform(data_test)
        #     pd_temp = (
        #         pd.DataFrame(df_train, columns=['set_P', 'SOC_start', 'SOC_end', 'set_SOH', 'temper', 'delta_SOH']))
        #     temp_trans = torch.from_numpy(
        #         pd_temp[['set_P', 'SOC_start', 'SOC_end', 'set_SOH', 'temper']].values.astype(np.float32))
        #
        #     # deg = deg_model(temp_trans).detach().numpy()[0]
        #     deg = deg_model(temp_trans)
        #
        #     ans = pd.DataFrame(deg.detach().numpy())
        #     ans_pro = pd.concat([pd_temp[['set_P', 'SOC_start', 'SOC_end', 'set_SOH', 'temper']], ans], axis=1)
        #     ans_soh = pd.DataFrame(zscores.inverse_transform(ans_pro)).iloc[:, -1]  # 得出最终的delta_SOH
        #     deg = ans_soh.values[0]
        #     # 根据老化量更新最大容量，以及SOH
        #     self.soh += deg
        #     self.ENERGY_MAX = (1 + deg) * self.ENERGY_MAX

        # use history cost
        if not is_evaluate:
            if pre_energy > 0:
                # self.step_reward = p_da_rescale[next_node][time_step + duration_time] * delta_energy - self.history_cost * delta_energy / pre_energy - self.DEG_COST * delta_energy - tra_cost
                # use pwr to compute cost
                self.step_reward = p_da_rescale[next_node][
                                       time_step + duration_time] * pwr * SCALE - self.history_cost * delta_energy / pre_energy - self.DEG_COST * pwr * SCALE - tra_cost
            else:
                self.step_reward = -self.DEG_COST * delta_energy - tra_cost
        else:
            # use RT price to compute reward
            if pre_energy > 0:
                # self.step_reward = p_da_rescale[next_node][time_step + duration_time] * delta_energy - self.history_cost * delta_energy / pre_energy - self.DEG_COST * delta_energy - tra_cost
                # use pwr to compute cost
                self.step_reward = p_rt_rescale[next_node][
                                       time_step + duration_time] * pwr * SCALE - self.history_cost * delta_energy / pre_energy - self.DEG_COST * pwr * SCALE - tra_cost
            else:
                self.step_reward = -self.DEG_COST * delta_energy - tra_cost

        if pre_energy > 0:
            self.history_cost -= self.history_cost * delta_energy / pre_energy
        else:
            self.history_cost = 0

    def hold(self, next_node, duration_time, SCALE):

        tra_cost = self.TRA_COST * duration_time * SCALE
        self.current_node = next_node

        self.step_reward = -1 * tra_cost
