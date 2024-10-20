#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Module bat_sta

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

from tqdm import tnrange, tqdm_notebook

def load_data(data_path):
    
    # pkl_file = open(data_path+'location_data.pkl', 'rb')
    # location_data = pickle.load(pkl_file)
    # pkl_file.close()
    # pkl_file = open(data_path+'node_id_set.pkl', 'rb')
    # node_id_set = pickle.load(pkl_file)
    # pkl_file.close()
    #
    # pkl_file = open(data_path+'RT_travel_time_best_region.pkl', 'rb')
    # duration_data = pickle.load(pkl_file)
    # pkl_file.close()
    # pkl_file = open(data_path+'Travel_distance_data.pkl', 'rb')
    # distance_data = pickle.load(pkl_file)
    # pkl_file.close()

    with open(data_path+'LMP_data.pkl', 'rb') as pkl_file:
        LMP_data = pickle.load(pkl_file)

    market_run_id = 'HASP'
#     market_run_id = 'RTPD'

    # pkl_file = open(data_path+'LMP_data_'+market_run_id+'.pkl', 'rb')
    pkl_file = open(data_path+'LMP_data_'+market_run_id+'_CAISO.pkl', 'rb')
    LMP_data_RTM = pickle.load(pkl_file)
    pkl_file.close()
#
#     pkl_file = open(data_path+'SR_price_data_DAM.pkl', 'rb')
#     price_data_SR = pickle.load(pkl_file)
#     pkl_file.close()
#
#     pkl_file = open(data_path+'NR_price_data_DAM.pkl', 'rb')
#     price_data_NR = pickle.load(pkl_file)
#     pkl_file.close()

    with open(data_path + 'node_data_clean.pkl', 'rb') as pkl_file:
        node_data_clean = pickle.load(pkl_file)
    
    print('Data loaded.')
    
    # return (location_data,node_id_set,duration_data,distance_data,LMP_data,LMP_data_RTM,price_data_SR,price_data_NR,node_data_clean)
    return (LMP_data, LMP_data_RTM, node_data_clean)

