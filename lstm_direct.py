#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time     : 2020/8/31 15:13
@Author   : ji hao ran
@File     : lstm_direct.py
@Project  : waterGame
@Software : PyCharm
"""
from utils import *

"""
lstm_direct策略（直接）
共需要建立k个模型，每个模型预测步为w=steps_out/k
[y(t)`,y(t+1)`,...,y(t+w-1)`,]=model_1[X(t-1),X(t-2),...,X(t-steps_in)]
[y(t+w)`,y(t+w+1)`,...,y(t+2w-1)`,]=model_2[X(t-1),X(t-2),...,X(t-steps_in)]
...
[y(t+(k-1)*w)`,y(t+(k-1)*w+1)`,...,y(t+k*w-1)`,]=model_k[X(t-1),X(t-2),...,X(t-steps_in)]


直接策略中，每个模型的输入都相同，分别建立不同的模型预测相同的步数，合并每次的预测结果，为一周多步的预测结果。
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
    # 直接预测步
    steps_out = [1, 4, 8]
    # 网格参数
    grid_df = expand_grid(targets=[['Qi']],
                          features=features,
                          steps_out=steps_out,
                          steps_in=[i * 3 for i in steps_out],
                          lr=[1e-3],
                          units=[50, 75, 100],
                          activation=['tanh'],
                          drop_out=[0],
                          epochs=[30],
                          batches=[32])

    # 网格训练，自动参数寻优
    output_path = 'e:/waterGame/result/direct/'
    grid_train(grid_df, data, output_path)
