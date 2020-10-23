#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time     : 2020/8/27 16:11
@Author   : ji hao ran
@File     : utils.py
@Project  : waterGame
@Software : PyCharm
"""
import numpy as np

np.random.seed(100)

import tensorflow as tf

tf.random.set_seed(200)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, RepeatVector, TimeDistributed, Bidirectional, Conv1D, MaxPooling1D, Flatten
from keras import optimizers
import itertools
import time
import os
import re


def data_preprocessing(parent_path='e:/waterGame/data/prepare/'):
    # 批量处理
    result = map(lambda x: pd.read_csv(parent_path + x, index_col=0, parse_dates=True), os.listdir(parent_path))
    # 返回
    return tuple(result)


# 超参数网格化
def expand_grid(**kwargs):
    name = list(kwargs.keys())
    product = list(itertools.product(*kwargs.values()))
    df = pd.DataFrame({name[i]: [x[i] for x in product] for i in range(len(kwargs))})
    result = pd.concat([pd.DataFrame(dict(id=range(1, len(df) + 1, 1))), df], axis=1)
    return result


# 分割序列为监督学习
def to_supervised(data, x, y, n_lag=1, n_seq=1, week_seq=56, drop_nan=True, remain_merge=False, valid=False):
    """时间序列转为监督学习数据

    :param data: Sequence of observations as a list or NumPy array.
    :param n_lag: Number of lag observations as input (X).
    :param n_seq: Number of observations as output (y).
    :param drop_nan: 删除特征的缺失值
    :param x:
    :param y:
    :param remain_merge:
    :param valid:
    :param week_seq:
    :return: Pandas DataFrame of series framed for supervised learning.
    """
    df = pd.DataFrame(data)

    # 滞后数据
    lag_df = df[x]
    # 推移数据
    seq_df = df[y]
    # 剩余数据
    remain_df = df[[i for i in df.columns if i not in x]]
    # 目标数据与特征数据初始化
    target, feature, target_names, feature_names = [], [], [], []
    # 目标数据 (t, t+1, ... t+n)
    for i in range(0, n_seq):
        target.append(seq_df.shift(-i))
        if i == 0:
            target_names += [('%s(t)' % j) for j in y]
        else:
            target_names += [('%s(t+%d)' % (j, i)) for j in y]
    # 特征数据 (t-n, ... t-1)
    for i in range(n_lag, 0, -1):
        feature.append(lag_df.shift(i))
        feature_names += [('%s(t-%d)' % (j, i)) for j in x]
    # 一周目标数据
    week_target, week_target_names = [], []
    for i in range(0, week_seq):
        week_target.append(seq_df.shift(-i))
        if i == 0:
            week_target_names += [('%s(T)' % j) for j in y]
        else:
            week_target_names += [('%s(T+%d)' % (j, i)) for j in y]
    agg = pd.concat(target + feature + week_target, axis=1)
    # 列名赋值
    agg.columns = target_names + feature_names + week_target_names
    # 合并其他特征
    if remain_merge:
        agg = agg.merge(remain_df, how='left', left_index=True, right_index=True)
        feature_names = feature_names + remain_df.columns.tolist()
    # 预测集x
    pred_x = agg[feature_names].dropna().tail(1)
    # 删除行缺失值
    if drop_nan:
        agg.dropna(inplace=True)
    # 分开
    if valid:
        return pred_x
    else:
        return agg[feature_names], agg[target_names], agg[week_target_names]


# 训练lstm模型
def fit_lstm(train_x, train_y, steps_in, steps_out, n_features, lr, units, activation, drop_out, epochs, batches):
    # 数据框转为数组
    x = np.array(train_x)
    y = np.array(train_y)

    # x变形 [samples, timesteps, features]
    # x shape [samples,steps_in,n_features] y shape [samples,steps_out]
    x = x.reshape((train_x.shape[0], steps_in, n_features))

    # 设计lstm模型，多维多步模型
    adam = optimizers.Adam(learning_rate=lr)  # 优化器
    model = Sequential()
    model.add(LSTM(units, activation=activation, input_shape=(steps_in, n_features), return_sequences=True))
    model.add(LSTM(units, activation=activation, dropout=drop_out))
    model.add(Dense(steps_out))
    model.compile(optimizer=adam, loss='mse')
    model.fit(x, y, epochs=epochs, batch_size=batches, verbose=2)

    return model


# 训练lstm模型
def fitting_lstm(train_x, train_y, valid_x, valid_y, steps_in, steps_out, n_features, lr, units, activation, drop_out,
                 epochs, batches, lstm_kind, shuffle):
    # 数据框转为数组
    x = np.array(train_x)
    y = np.array(train_y)
    valid_x = np.array(valid_x)
    valid_y = np.array(valid_y)

    # x变形 [samples, timesteps, features]
    x = x.reshape((train_x.shape[0], steps_in, n_features))
    valid_x = valid_x.reshape((valid_x.shape[0], steps_in, n_features))

    # 模型初始化
    model = Sequential()
    # 设计不同的模型
    if lstm_kind == 'vector':
        model.add(LSTM(units, activation=activation, input_shape=(steps_in, n_features), return_sequences=True))
        model.add(LSTM(units, activation=activation, dropout=drop_out))
        model.add(Dense(steps_out))
    elif lstm_kind == 'encoder-decoder':
        # y变形[samples,timesteps,features]
        y = y.reshape((train_y.shape[0], steps_out, 1))
        # encoder-decoder lstm
        model.add(LSTM(units, activation=activation, input_shape=(steps_in, n_features)))  # encoder
        model.add(RepeatVector(steps_out))  # decoder
        model.add(LSTM(units, activation=activation, return_sequences=True))  # decoder
        model.add(TimeDistributed(Dense(1, activation=activation)))
        model.add(TimeDistributed(Dense(1)))
    elif lstm_kind == 'bidirectional':
        # 双向lstm
        model.add(Bidirectional(LSTM(units, activation=activation), input_shape=(steps_in, n_features)))
        model.add(Dense(steps_out))
    else:
        # 堆叠lstm
        model.add(LSTM(units, activation=activation, return_sequences=True, input_shape=(steps_in, n_features)))
        model.add(LSTM(units, activation=activation, return_sequences=True, dropout=drop_out, recurrent_dropout=0.2))
        model.add(LSTM(units, activation=activation, dropout=drop_out, recurrent_dropout=0.2))
        model.add(Dense(steps_out))
    # 优化器
    adam = optimizers.Adam(learning_rate=lr)
    # 模型编译
    model.compile(optimizer=adam, loss='mse')
    model.fit(x, y, epochs=epochs, batch_size=batches, verbose=2, shuffle=shuffle, validation_data=(valid_x, valid_y))

    return model


# 指标计算(nse,mse)
def cal_metric(actual, predict, w1_samples: float = 2 / 7, w1: float = 0.65, w2: float = 0.35):
    """
    加权平均纳什效率系数NSE
    NSE越接近1，表示模型质量好，可信度高；
    NSE接近0，表示模拟结果接近观测值的平均水平，即总体结果可信，但过程模拟误差大；
    NSE远小于0，表示模型不可信

    :param predict: 预测值,array,2d
    :param actual: 实际值,array,2d
    :param w1_samples: w1权重的样本占比
    :param w1: w1权重系数
    :param w2: w2权重系数
    :return: NSE,MSE
    """

    def one_dim_nse(y_true, y_pred):
        s = int(len(y_pred) * w1_samples)
        t1 = w1 * np.sum((y_true[:s] - y_pred[:s]) ** 2) / np.sum((y_true[:s] - np.mean(y_true)) ** 2)
        t2 = w2 * np.sum((y_true[s:-1] - y_pred[s:-1]) ** 2) / np.sum((y_true[s:-1] - np.mean(y_true)) ** 2)
        # 返回nse
        return 1 - t1 - t2

    # 每行计算nse
    nse = list(map(one_dim_nse, actual, predict))
    # 每行计算mse
    mse = list(map(mean_squared_error, actual, predict))
    # 每列计算mse
    col_mse = list(map(mean_squared_error, actual.T, predict.T))
    return nse, mse, col_mse


def week_predict(model, X, steps_in, steps_out, n_features):
    if steps_out == 56:
        # x变形
        X = X.reshape((X.shape[0], steps_in, n_features))
        # 预测值
        y_week_predict = model.predict(X)
        # 3维变2维
        if y_week_predict.shape.__len__() == 3:
            y_week_predict = y_week_predict.reshape((y_week_predict.shape[0], y_week_predict.shape[1]))
    else:
        y_week_window = list()
        X_append = X
        for i in range(int(np.ceil(56 / steps_out))):
            # 滑动窗口取x
            X_window = X_append[:, -steps_in:]
            # x变形
            X_window = X_window.reshape((X_window.shape[0], steps_in, n_features))
            # 预测
            y_window = model.predict(X_window)
            # 3维变2维
            if y_window.shape.__len__() == 3:
                y_window = y_window.reshape((y_window.shape[0], y_window.shape[1]))
            # 保存结果
            y_week_window.append(y_window)
            # 预测结果放入X
            X_append = np.column_stack((X_append, y_window))
        y_week_predict = np.column_stack(y_week_window)[:, :56]
    return y_week_predict


# 模型评估
def week_evaluate_model(seq, output_path, model, x_df, y_df, y_week_df, steps_in, steps_out, n_features, prefix=None):
    """评估模型在训练集，验证集上一周预测的效果

    :param model: lstm model
    :param x_df:
    :param y_df:
    :param y_week_df:
    :param steps_in:
    :param steps_out:
    :param n_features:
    :param prefix:
    :return:
    """
    # 转为数组
    X = np.array(x_df)
    y = np.array(y_df)
    y_week_actual = np.array(y_week_df)
    # 周预测值
    y_week_predict = week_predict(model, X, steps_in, steps_out, n_features)
    # 周指标计算
    nse, mse, col_mse = cal_metric(actual=y_week_actual, predict=y_week_predict)
    # 横向指标数据框
    row_df = pd.DataFrame({'NSE': nse, 'MSE': mse}, index=y_week_df.index)
    # 周真实值与预测值数据
    week_df = pd.DataFrame(np.column_stack((y_week_actual, y_week_predict)), index=y_week_df.index)
    week_df.columns = np.append(y_week_df.columns + '_actual', y_week_df.columns + '_predict')
    pd.concat([week_df, row_df], axis=1).to_csv(output_path + 'grid_' + str(seq) + '_' + prefix + '.csv')
    # 纵向指标数据框
    col_df = pd.DataFrame({'CMSE': col_mse}, index=y_week_df.columns)
    # 指标统计量,反映模型性能
    stat_df = row_df.agg(['mean']).melt(ignore_index=False)
    # 变形为一行
    stat_df = stat_df[['value']]. \
        set_index(prefix + '_' + stat_df['variable']).T
    # 返回统计指标、数据框(画图)
    return stat_df, (row_df, col_df)


# 指标绘图
def metric_plot(seq, train_detail, valid_detail, output_path):
    train_row, train_col = train_detail
    valid_row, valid_col = valid_detail
    # 1 行指标图
    plt.switch_backend('agg')
    plt.figure(figsize=(20, 12))
    for i, (data, name) in enumerate([(valid_row, 'valid'), (train_row, 'train')]):
        for j, col in enumerate(['NSE', 'MSE']):
            plt.subplot(221 + 2 * i + j)
            plt.plot(data[col], color='r' if col == 'MSE' else 'g', label=col)
            plt.legend()
            plt.title(name + col)
    plt.savefig(output_path + 'grid_' + str(seq) + '_nse.png')

    # 2 列指标图
    r = 90  # x轴刻度旋转角度（逆时针）
    plt.figure(figsize=(18, 12))
    for i, (data, name) in enumerate([(valid_col, 'valid'), (train_col, 'train')]):
        plt.subplot(211 + i)
        plt.plot(data, label='CMSE')
        plt.legend()
        exec('plt.xticks([])' if name != 'train' else 'plt.xticks(rotation=r)')
        plt.title(name + 'CMSE')
    plt.savefig(output_path + 'grid_' + str(seq) + '_mse.png')
    plt.close()


# 单次训练
def single_train(data, row_para, output_path):
    # 数据与参数
    train_set, train_set34, train_set56, train_set567, train_set7, valid_set1, valid_set2, valid_set3 = data
    paras = row_para.iloc[0, :].to_dict().values()
    seq, targets, features, steps_out, steps_in, lr, units, activation, drop_out, epochs, batches, lstm_kind, train_valid_kind = paras

    # step 1: 时间序列数据转为监督学习数据

    # 训练数据
    train_x, train_y, train_y_week = to_supervised(train_set, features, targets, steps_in, steps_out)
    train34_x, train34_y, train34_y_week = to_supervised(train_set34, features, targets, steps_in, steps_out)
    train56_x, train56_y, train56_y_week = to_supervised(train_set56, features, targets, steps_in, steps_out)
    train7_x, train7_y, train7_y_week = to_supervised(train_set7, features, targets, steps_in, steps_out)
    train567_x, train567_y, train567_y_week = to_supervised(train_set567, features, targets, steps_in, steps_out)
    # 预测数据
    test1_x = to_supervised(valid_set1, features, targets, steps_in, steps_out, valid=True)
    test2_x = to_supervised(valid_set2, features, targets, steps_in, steps_out, valid=True)
    test3_x = to_supervised(valid_set3, features, targets, steps_in, steps_out, valid=True)
    # 选择训练数据，验证数据
    if train_valid_kind == 'all-7v':
        train_x = train_x
        train_y = train_y
        train_y_week = train_y_week
        valid_x = train7_x
        valid_y = train7_y
        valid_y_week = train7_y_week
        shuffle = True
    elif train_valid_kind == '56t-7v':
        train_x = train56_x
        train_y = train56_y
        train_y_week = train56_y_week
        valid_x = train7_x
        valid_y = train7_y
        valid_y_week = train7_y_week
        shuffle = False
    elif train_valid_kind == '567t-7v':
        train_x = train567_x
        train_y = train567_y
        train_y_week = train567_y_week
        valid_x = train7_x
        valid_y = train7_y
        valid_y_week = train7_y_week
        shuffle = False
    else:
        train_x = train567_x
        train_y = train567_y
        train_y_week = train567_y_week
        valid_x = train34_x
        valid_y = train34_y
        valid_y_week = train34_y_week
        shuffle = False

    # step 2: 训练模型

    # 特征个数
    n_features = len(features)
    # lstm模型
    model = fitting_lstm(train_x, train_y, valid_x, valid_y, steps_in, steps_out, n_features, lr, units,
                         activation, drop_out, epochs, batches, lstm_kind, shuffle)
    # step 3：模型评估

    # 对训练数据、验证数据评估
    train_stat, train_detail = week_evaluate_model(seq, output_path, model, train_x, train_y, train_y_week, steps_in,
                                                   steps_out, n_features, 'train')
    valid_stat, valid_detail = week_evaluate_model(seq, output_path, model, valid_x, valid_y, valid_y_week, steps_in,
                                                   steps_out, n_features, 'valid')

    # 评估汇总（训练，验证）
    row_stat = pd.concat([train_stat, valid_stat], axis=1). \
        set_axis(row_para.index). \
        assign(score1=0, score2=0, score3=0, score=0). \
        round(4)
    # 合并参数与评估
    metric = pd.concat([row_para, row_stat], axis=1)
    # 输出，追加模式
    metric.to_csv(output_path + 'metric.csv', index=False, mode='a', header=True if seq == 1 else False)

    # step 4:评估可视化

    metric_plot(seq, train_detail, valid_detail, output_path)

    # step 5:预测

    submission(seq, model, (test1_x, test2_x, test3_x), row_para, output_path)

    # 输出评价指标，模型，预测集x
    return metric, model, (test1_x, test2_x, test3_x)


# step 2:网格训练
def grid_train(grid_df, data, output_path):
    t_s = time.time()

    # step 1:遍历网格中的参数
    grid_metric, grid_model, grid_test = list(), list(), list()
    for i in range(len(grid_df)):
        try:
            # if i < 11:
            #     continue
            # 打印头信息
            print('#' * 40)
            print('第{}组网格参数训练开始!'.format(i + 1), '进度{}/{},{:.1%}'.format(i + 1, len(grid_df), (i + 1) / len(grid_df)))
            t1 = time.time()

            # 训练模型，输出结果

            tf.random.set_seed(1)
            metric, model, test = single_train(data, grid_df.iloc[[i], :], output_path)

            # 结果存入列表
            grid_metric.append(metric)
            grid_model.append(model)
            grid_test.append(test)

            # 打印尾信息
            t2 = time.time()
            print('第{}组网格参数训练完成！'.format(i + 1), '用时{:.1f}分钟，训练结果为：'.format((t2 - t1) / 60))
            print(metric.T)

        except:
            continue

    # step 2:合并评价指标
    grid_metric_df = pd.concat(grid_metric).reset_index(drop=True)

    # step 3:依据得分寻最优模型
    # best_index = grid_metric_df['NSE'].idxmax()

    # step 4:输出
    # best_para = grid_df.iloc[best_index, :]
    # best_model = grid_model[best_index]
    # best_test = grid_test[best_index]
    t_e = time.time()
    print('网格训练总用时{:.0f}分钟'.format((t_e - t_s) / 60))
    # grid_metric_df.to_csv(output_path + 'metric.csv', index=False)
    # return best_model, best_test, best_para


# 预测
def make_predict(model, test_x, steps_in, steps_out, n_features, seq_name=None):
    y_week_predict = week_predict(model, test_x.values, steps_in, steps_out, n_features)
    test_df = pd.DataFrame(y_week_predict, columns=['Prediction' + str(i) for i in range(1, 57, 1)])
    seq_df = pd.DataFrame(dict(SeqNum=seq_name), index=[0])
    return pd.concat([seq_df, test_df], axis=1)


# step 3: 预测测试数据输出
def submission(seq, model, test, para, output_path):
    # 最优参数
    best = para.iloc[0, :].to_dict()
    steps_in, steps_out, n_features = best.get('steps_in'), best.get('steps_out'), len(best.get('features'))

    # 最优测试集
    test1_x, test2_x, test3_x = test

    # 预测
    test1_predict = make_predict(model, test1_x, steps_in, steps_out, n_features, seq_name='1')
    test2_predict = make_predict(model, test2_x, steps_in, steps_out, n_features, seq_name='2')
    test3_predict = make_predict(model, test3_x, steps_in, steps_out, n_features, seq_name='3')

    # 结果输出
    mission = pd.concat([test1_predict, test2_predict, test3_predict])
    mission.to_csv(output_path + 'grid_' + str(seq) + '_submission.csv', index=False)
    print('预测完成！')
