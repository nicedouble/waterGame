#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time     : 2020/9/1 11:39
@Author   : ji hao ran
@File     : lstm_mimo.py
@Project  : waterGame
@Software : PyCharm
"""

from utils import *

"""
lstm_mimo策略（多输入，多输出）
[y(t)`,y(t+1)`,...,y(t+steps_out-1)`]=model[X(t-1),X(t-2),...,X(t-steps_in)]

利用历史特征，直接完成一周多步的预测，只需要建立一个LSTM模型
模型输入X形状为3维：[samples,steps_in,n_features]
模型输出y形状为2维：[samples,steps_out]

模型mse与预测步正相关，步越大，越预测不准
特征滞后步一般是3倍预测步周期，效果较好

"""
if __name__ == '__main__':
    # step 1: 数据准备
    data = data_preprocessing()

    # step 2:模型网格训练
    # 特征正则表达式
    regex_features = [
        'Qi',
        'Qi|T', 'Qi|w$', 'Qi|R.*', 'Qi|month.*', 'Qi|season.*', 'Qi|D.*',
        'Qi|T|R.*', 'Qi|T|w$', 'Qi|T|month.*', 'Qi|T|season.*', 'Qi|T|D.*',
        'Qi|T|R.*|w$', 'Qi|T|R.*|season.*', 'Qi|T|R.*|month.*',
        'Qi|T|R.*|season.*|month.*',
    ]
    # 特征名
    features = [[j[0] for j in [re.findall(k, i) for i in data[0].columns] if j] for k in regex_features]
    # 网格参数
    grid_df = expand_grid(targets=[['Qi']],
                          features=features,
                          steps_out=[56],
                          steps_in=[i * 56 for i in [3]],
                          lr=[1e-3],
                          units=[50, 75, 100],
                          activation=['tanh'],
                          drop_out=[0],
                          epochs=[30],
                          batches=[32])

    # 网格训练，自动参数寻优
    output_path = 'e:/waterGame/result/mimo/'
    # best_model, best_test, best_para = grid_train(grid_df, data, output_path)
    grid_train(grid_df, data, output_path)

    # step 3: 预测输出
    # submission(best_model, best_test, best_para, output_path)
