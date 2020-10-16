#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time     : 2020/9/2 15:57
@Author   : ji hao ran
@File     : lstm_recursive.py
@Project  : waterGame
@Software : PyCharm
"""

from utils import *

"""
lstm_recursive策略（递归）
y(t)`=model[y(t-1),y(t-2),...,y(t-steps_in)]
y(t+1)`=model[y(t)`,y(t-1),y(t-2),...,y(t-steps_in-1)]
...
y(t+steps_out-1)`=model[y(t+steps_out-2)`,...,y(t)`,y(t-1),y(t-2),...,y(t-steps_in-steps_out+1)]

递归只需要建立一个模型，只能用y预测y,将上次预测结果作为下次预测的输入，不断向前滚动预测，最终完成一周多步的预测。
"""
if __name__ == '__main__':
    # step 1: 数据准备
    data = data_preprocessing()

    # step 2:模型网格训练
    # 递归预测步
    steps_out = [1, 8, 16, 24]
    # 定义网格参数
    grid_df = expand_grid(targets=[['Qi']],
                          features=[['Qi']],
                          steps_out=steps_out,
                          steps_in=[i * 3 for i in steps_out],
                          lr=[1e-3],
                          units=[100],
                          activation=['tanh'],
                          drop_out=[0],
                          epochs=[30],
                          batches=[32])

    # 网格训练
    output_path = 'e:/waterGame/result/recursive/'
    grid_train(grid_df, data, output_path)
