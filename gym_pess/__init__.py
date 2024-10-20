# coding=utf-8
"""
Author:DYK
Email:y.d@pku.edu.cn

date:27/6/2022 下午4:47
desc:
"""
from gym.envs.registration import register

register(
    id='PESS-v0',
    entry_point='gym_pess.envs:PESSEnv',
)