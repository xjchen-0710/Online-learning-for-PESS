# coding=utf-8
"""
Author: DYK
Email: y.d@pku.edu.cn

Date: 2023/2/8 22:18
Desc: 
"""

import torch.nn as nn
import torch.nn.functional as F

class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression, self).__init__()
        self.hidden1 = nn.Linear(in_features=5, out_features=20, bias=True)
        #         self.hidden2 = nn.Linear(200, 100)
        #         self.hidden3 = nn.Linear(100, 50)
        # 回归预测层
        self.predict = nn.Linear(20, 1)

    # 定义网络前向传播路径
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        #         x = F.relu(self.hidden2(x))
        #         x = F.relu(self.hidden3(x))
        output = self.predict(x)
        # 输出一个一维向量
        return output[:, 0]