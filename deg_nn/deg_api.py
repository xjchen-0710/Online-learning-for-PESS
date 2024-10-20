# coding=utf-8
"""
Author: DYK
Email: y.d@pku.edu.cn

Date: 2023/2/8 15:13
Desc: 
"""
import pandas as pd
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sklearn
from deg_nn.mlp import MLPregression
import __main__
setattr(__main__, "MLPregression", MLPregression)

# class MLPregression(nn.Module):
#     def __init__(self):
#         super(MLPregression, self).__init__()
#         # 第一个隐含层
#         self.hidden1 = nn.Linear(in_features=5, out_features=200, bias=True)
#         # 第二个隐含层
#         self.hidden2 = nn.Linear(200, 100)
#         # 第三个隐含层
#         self.hidden3 = nn.Linear(100, 50)
#         # 回归预测层
#         self.predict = nn.Linear(50, 1)
#
#     # 定义网络前向传播路径
#     def forward(self, x):
#         x = F.relu(self.hidden1(x))
#         x = F.relu(self.hidden2(x))
#         x = F.relu(self.hidden3(x))
#         output = self.predict(x)
#         # 输出一个一维向量
#         return output[:, 0]

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



# standard = pickle.load(open('torch_standardscaler_best.pkl', 'rb'))
# model = torch.load('torch_model_best.pth')
# temp_list = [[250,0,1,1,25,0] for i in range(350)] # 前提 在输入时随便设定一个delta_SOH  可以是0 使标准化的维度统一
# for i in range(0,350):
#     temp_list[i][0] = (i)
# temp = pd.DataFrame(temp_list)
# temp_trans = standard.transform(temp) # 全数据标准化
# pd_temp = (pd.DataFrame(temp_trans,columns = ['set_P','SOC_start','SOC_end','set_SOH','temper','delta_SOH']))
# temp_trans = torch.from_numpy(pd_temp[['set_P','SOC_start','SOC_end','set_SOH','temper']].values.astype(np.float32))
#
# ans = model(temp_trans)
# ans = pd.DataFrame(ans.detach().numpy())
# ans_pro = pd.concat([pd_temp[['set_P','SOC_start','SOC_end','set_SOH','temper']], ans], axis = 1)
# ans_soh = pd.DataFrame(standard.inverse_transform(ans_pro)).iloc[:,-1] # 得出最终的delta_SOH
# print(ans_soh)




# data_test = pd.DataFrame([[27000,1,0.15,1-0.0011,25]],columns = ['set_P','SOC_start','SOC_end','set_SOH','temper'])
zscores = pickle.load(open('torch_standardscaler_best3.pkl', 'rb'))
# df_train = zscores.transform(data_test)

model = MLPregression()
model = torch.load('torch_model_best3.pth', map_location='cpu')#TODO 补全 加载
# df_train = torch.from_numpy(df_train.astype(np.float32))
# ans = model(df_train)
# print(ans)
