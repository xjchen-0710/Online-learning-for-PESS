#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division
import pandas as pd
import requests, zipfile, io
import pytz, datetime
from io import BytesIO

from datetime import timedelta
import numpy as np
import time
import json
import os
import pickle
# import xmltodict
# import polyline


from pprint import pprint
from pyomo.environ import *


import pandas as pd

import matplotlib.pyplot as plt
import cloudpickle
from traffic_data_fetch import *
from node_distance_cal import *

from tqdm import tnrange, tqdm_notebook
from IPython.display import clear_output
from geopy.distance import geodesic

def load_node_set_rect(lat_u,lat_l,lon_u,lon_l,node_data):
## Define Region
    # lat_u = location_data['COLNGA2_6_N001']['location'][0]
    # lat_l = location_data['DEVLSDN_6_N001']['location'][0]
    # lon_u = location_data['DEVLSDN_6_N001']['location'][1]
    # lon_l = location_data['Q633C1_7_N001']['location'][1]
    node_name_list = node_data['node_name']
    node_set = []
    for node_i in node_name_list:
        node_lat = float(node_data[node_data['node_name'] == node_i]['lat'])
        node_lon = float(node_data[node_data['node_name'] == node_i]['lon'])
        flag = node_lat>=lat_l and node_lat<=lat_u and node_lon>=lon_l and node_lon<=lon_u
        if flag:
            node_set.append(node_i)   
    return(node_set)

def load_node_set_circle(center_node,radius,node_data):
    node_set = []
    node_name_list = node_data['node_name']
    for node_i in node_name_list:
        flag = node_distance_cal(center_node, node_i, node_data) <= radius
        if flag:
            node_set.append(node_i)   
    return(node_set)

def travel_data_cal(node_set,node_data):
    speed = 35
    distance_cal_factor = 2.2
    duration_cal_factor = 1.3    
    node_num = len(node_set) 
    duration = np.zeros((node_num,node_num))
    distance = np.zeros((node_num,node_num))

    count_i = 0
    for i in range(node_num):
        node_i = node_set[i]
        count_i = count_i + 1
#         clear_output()
#         print(count_i)
        for j in range(node_num):  
            node_j = node_set[j]
    #         count_j = count_j + 1
            if i!=j:
                distance_i_j = node_distance_cal(node_i,node_j,node_data)
                distance[i][j] = distance_i_j
                distance[j][i] = distance_i_j
                duration[j][i] = distance_i_j/speed*60
                duration[i][j] = distance_i_j/speed*60
    distance = distance * distance_cal_factor
    duration = duration * duration_cal_factor
    print('Travel data calcualted.')
    return(duration,distance)

def travel_data_fetch(node_set,location_data):
    speed = 35
    distance_cal_factor = 2.2
    duration_cal_factor = 1.3    
    print_error = 1
    node_num = len(node_set) 
    duration = np.zeros((node_num,node_num))
    distance = np.zeros((node_num,node_num))
    print('Fetching data...')
    for node_a in tqdm_notebook(range(node_num)):
        for node_b in tqdm_notebook(range(node_num)):
            if node_b>node_a:
                node_set_temp = [node_set[node_a],node_set[node_b]]
                success_flag = 0
                while success_flag == 0:                        
                    success_flag = 1                        
                    try:
                        [duration_temp,distance_temp] = travel_matrix_fetch(node_set_temp,location_data)
                    except Exception as e:
                        success_flag = 0
                        if print_error == 1:
                            print('Error:',e)
                            print('Retring...','Node_set:',node_set_temp)
                        time.sleep( 5 )
                        duration[node_a][node_b] = duration_temp[0][1]
                        duration[node_b][node_a] = duration_temp[1][0]
                        distance[node_a][node_b] = distance_temp[0][1]
                        distance[node_b][node_a] = distance_temp[1][0]                                

    print('Travel data fetched')
    duration = np.asarray(duration)
    distance = np.asarray(distance)
    return(duration,distance)


def node_distance_cal(node_i, node_j, df):
    start_lat = float(df[df['node_name'] == node_i]['lat'])
    start_lon = float(df[df['node_name'] == node_i]['lon'])
    end_lat = float(df[df['node_name'] == node_j]['lat'])
    end_lon = float(df[df['node_name'] == node_j]['lon'])
    distance = geodesic((start_lat, start_lon), (end_lat, end_lon)).miles
    return (distance)