from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime, timedelta
import pandas as pd
import math
import numpy as np
import random
from tqdm import trange

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from math import sqrt
from pandas import read_csv, DataFrame
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#elect 기준으로 계산해보면 32304,370인 데이터에서
#window_size 192, stride_size 24 (디폴트) 로 설정하면
#(32304-192)//24 을 수행하므로 1339개의 윈도우가 나오게 되는데
#370개의 time-series가 있으므로 1339*370을 하면 495430이 나옴.
#그런데 train_data를 확인해보면 389101개임.
#이 이유는 data_start를 설정해주는 과정에서 처음으로 0이 나오지 않는 지점을 훈련데이터의 시작지점으로 삼아서인듯.
#대략 확인해보면 8760 정도에서 처음으로 0이 아닌 데이터가 나오는 시계열이 약 180개.
#대충 계산해보면 맞는 수치로 확인됨

def prep_data(data, covariates, data_start, train = True): # covariate는 전체 length에 대한 배열
    """Divide the training sequence into windows"""
    time_len = data.shape[0] # 32304
    input_size = window_size-stride_size # 192 - 24 = 168
    windows_per_series = np.full((num_series), (time_len-input_size) // stride_size)#370, (32304-128)//24
    if train: windows_per_series -= (data_start+stride_size-1) // stride_size
    total_windows = np.sum(windows_per_series) # total_windows = 389101
    x_input = np.zeros((total_windows, window_size, 1 + num_covariates + 1), dtype='float32') 
    label = np.zeros((total_windows, window_size), dtype='float32')
    v_input = np.zeros((total_windows, 2), dtype='float32')
    count = 0
    if not train:
        covariates = covariates[-time_len:]
    for series in trange(num_series):
        cov_age = stats.zscore(np.arange(total_time-data_start[series])) # shape:(series_len,)
        if train:
            covariates[data_start[series]:time_len, 0] = cov_age[:time_len-data_start[series]] # positional encoding
        else:
            covariates[:, 0] = cov_age[-time_len:]
        for i in range(windows_per_series[series]):
            if train:
                window_start = stride_size*i+data_start[series]
            else:
                window_start = stride_size*i
            window_end = window_start+window_size
            '''
            print("x: ", x_input[count, 1:, 0].shape)
            print("window start: ", window_start)
            print("window end: ", window_end)
            print("data: ", data.shape)
            print("d: ", data[window_start:window_end-1, series].shape)
            '''
            x_input[count, 1:, 0] = data[window_start:window_end-1, series]
            x_input[count, :, 1:1+num_covariates] = covariates[window_start:window_end, :]
            x_input[count, :, -1] = series
            label[count, :] = data[window_start:window_end, series]
            nonzero_sum = (x_input[count, 1:input_size, 0]!=0).sum()
            if nonzero_sum == 0:
                v_input[count, 0] = 0
            else:
                v_input[count, 0] = np.true_divide(x_input[count, 1:input_size, 0].sum(),nonzero_sum)+1
                x_input[count, :, 0] = x_input[count, :, 0]/v_input[count, 0]
                if train:
                    label[count, :] = label[count, :]/v_input[count, 0]
            count += 1
    prefix = os.path.join(save_path, 'train_' if train else 'test_')
    np.save(prefix+'data_'+save_name, x_input)
    np.save(prefix+'v_'+save_name, v_input)
    np.save(prefix+'label_'+save_name, label)

def gen_covariates(times, num_covariates):
    """Get covariates"""
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 1] = input_time.weekday()
        covariates[i, 2] = input_time.hour
        covariates[i, 3] = input_time.month
    for i in range(1,num_covariates):
        covariates[:,i] = stats.zscore(covariates[:,i])
    return covariates[:, :num_covariates]

def visualize(data, week_start):
    x = np.arange(window_size)
    f = plt.figure()
    plt.plot(x, data[week_start:week_start+window_size], color='b')
    f.savefig("visual.png")
    plt.close()

if __name__ == '__main__':

    global save_path
    csv_path = 'data/LD2011_2014.txt'
    save_name = 'elect'
    window_size = 192 #192
    stride_size = 24 
    num_covariates = 4
    train_start = '2011-01-01 00:00:00'
    train_end = '2014-08-31 23:00:00'
    test_start = '2014-08-25 00:00:00' #need additional 7 days as given info
    test_end = '2014-09-07 23:00:00'
    pred_days = 7
    given_days = 7

    save_path = os.path.join('data', save_name)

    data_frame = pd.read_csv(csv_path, sep=";", index_col=0, parse_dates=True, decimal=',')
    data_frame = data_frame.resample('1h',label = 'left',closed = 'right').sum()[train_start:test_end] # hourly data로 변환
    data_frame.fillna(0, inplace=True)
    covariates = gen_covariates(data_frame[train_start:test_end].index, num_covariates) # shape : (32304,4)
    train_data = data_frame[train_start:train_end].values # shape: [seq_length, user_num] :  (32136,370)
    test_data = data_frame[test_start:test_end].values # shape : (336,370)
    data_start = (train_data!=0).argmax(axis=0) #find first nonzero value in each time series (370,0)
    total_time = data_frame.shape[0] # total seq_length : 32304
    
    num_series = data_frame.shape[1] #370은 시계열 인스턴스 갯수
    prep_data(train_data, covariates, data_start)
    prep_data(test_data, covariates, data_start, train=False)

    # train_data_elct.npy : 389101,192,6
    # train_label_elect.npy : 389101,192
    # train_v_elect.npy : 389101,2
    # test는 389101 이 아닌 2590
    # train 길이가 389101이기 때문에 훈련시키면 batch_size 8일때 48673개의 배치가 나오게 됨
