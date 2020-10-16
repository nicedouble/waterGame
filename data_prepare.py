#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time     : 2019/12/6 9:48
@Author   : ji hao ran
@File     : data_prepare.py
@Software : PyCharm
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# 数据分割函数，训练集、验证集
def split_data(data, train_fill=True, predict_fill=False):
    # 标准时间索引
    # 训练集
    index_train = pd.date_range(pd.Timestamp('20130101 0200'), pd.Timestamp('20180101'), freq='3h')
    # 验证集
    index_v1 = pd.date_range(pd.Timestamp('20180101 0200'), pd.Timestamp('20180201'), freq='3h')
    index_v2 = pd.date_range(pd.Timestamp('20180701 0200'), pd.Timestamp('20180801'), freq='3h')
    index_v3 = pd.date_range(pd.Timestamp('20181001 0200'), pd.Timestamp('20181101'), freq='3h')
    # 预测集
    index_p1 = pd.date_range(pd.Timestamp('20180201 0200'), pd.Timestamp('20180208'), freq='3h')
    index_p2 = pd.date_range(pd.Timestamp('20180801 0200'), pd.Timestamp('20180808'), freq='3h')
    index_p3 = pd.date_range(pd.Timestamp('20181101 0200'), pd.Timestamp('20181108'), freq='3h')

    # 左连接
    def left_merge(x, y):
        return pd.DataFrame(index=x).merge(y, how='left', right_index=True, left_index=True)

    # 训练
    if train_fill:
        train = left_merge(index_train, data[data.index.map(lambda x: x.year <= 2017)]).fillna(method='ffill')
    else:
        train = left_merge(index_train, data[data.index.map(lambda x: x.year <= 2017)])
    # 验证
    valid_data = data[data.index.map(lambda x: x.year == 2018)]
    v1 = left_merge(index_v1, valid_data[valid_data.index.map(lambda x: x.month == 1)]).fillna(method='ffill')
    v2 = left_merge(index_v2, valid_data[valid_data.index.map(lambda x: x.month == 7)]).fillna(method='ffill')
    v3 = left_merge(index_v3, valid_data[valid_data.index.map(lambda x: x.month == 10)]).fillna(method='ffill')
    # 预测
    if predict_fill:
        p1 = left_merge(index_p1, valid_data[valid_data.index.map(lambda x: x.month == 2)]).fillna(method='ffill')
        p2 = left_merge(index_p2, valid_data[valid_data.index.map(lambda x: x.month == 8)]).fillna(method='ffill')
        p3 = left_merge(index_p3, valid_data[valid_data.index.map(lambda x: x.month == 11)]).fillna(method='ffill')
    else:
        p1 = left_merge(index_p1, valid_data[valid_data.index.map(lambda x: x.month == 2)])
        p2 = left_merge(index_p2, valid_data[valid_data.index.map(lambda x: x.month == 8)])
        p3 = left_merge(index_p3, valid_data[valid_data.index.map(lambda x: x.month == 11)])
    # 合并
    valid1, valid2, valid3 = pd.concat([v1, p1]), pd.concat([v2, p2]), pd.concat([v3, p3])
    return train, valid1, valid2, valid3


# 数据准备
def prepare_data(parent_path='e:/waterGame/data/'):
    # 1 流量数据(3hour)
    flow_data = pd.read_excel(parent_path + 'one/入库流量数据.xlsx', index_col=0)
    # 分割
    flow_train, flow_valid1, flow_valid2, flow_valid3 = split_data(flow_data, train_fill=False)

    # 2 环境数据(1day)
    env_data = pd.read_excel(parent_path + 'one/环境表.xlsx', index_col=0, parse_dates=True, dtype={'wd': 'str'})
    env_data.index = env_data.index + pd.DateOffset(hours=2)
    # 风向one-hot编码
    env_data = pd.get_dummies(env_data, columns=['wd'])
    # 分割
    env_train, env_valid1, env_valid2, env_valid3 = split_data(env_data)

    # 3 降雨预报数据(1day)
    water_data = pd.read_excel(parent_path + 'one/降雨预报数据.xlsx', index_col=0)
    water_data.index = water_data.index + pd.DateOffset(hours=2)
    # 分割
    water_train, water_valid1, water_valid2, water_valid3 = split_data(water_data, predict_fill=True)

    # 4 遥测站降雨数据(1hour)，所有站点统计变换
    station_data = pd.read_excel(parent_path + 'one/遥测站降雨数据.xlsx', index_col=0)
    # 增加求和列
    station_data = station_data.assign(Rsum=lambda x: x.apply(np.sum, axis=1))
    # 归一化
    scaler = MinMaxScaler()
    station_data.iloc[:, :] = scaler.fit_transform(station_data)
    # 分割
    station_train, station_valid1, station_valid2, station_valid3 = split_data(station_data)

    # 5 增加时间属性
    time_data = pd.DataFrame(index=pd.date_range(pd.Timestamp('20130101 0200'), pd.Timestamp('20181108'), freq='3h')). \
        assign(month=lambda x: x.index.month, season=lambda x: [(i % 12 + 3) // 3 for i in x.index.month])
    # # one-hot编码
    time_data = pd.get_dummies(time_data, columns=['month', 'season'])
    # # 分割
    time_train, time_valid1, time_valid2, time_valid3 = split_data(time_data)

    # 6 数据分段合并
    train_set = pd.concat([flow_train, env_train, station_train, water_train, time_train], axis=1)
    valid_set1 = pd.concat([flow_valid1, env_valid1, station_valid1, water_valid1, time_valid1], axis=1)
    valid_set2 = pd.concat([flow_valid2, env_valid2, station_valid2, water_valid2, time_valid2], axis=1)
    valid_set3 = pd.concat([flow_valid3, env_valid3, station_valid3, water_valid3, time_valid3], axis=1)

    train_set.to_csv(parent_path + 'prepare/train_set.csv')
    valid_set1.to_csv(parent_path + 'prepare/valid_set1.csv')
    valid_set2.to_csv(parent_path + 'prepare/valid_set2.csv')
    valid_set3.to_csv(parent_path + 'prepare/valid_set3.csv')
    print('数据准备完成')


if __name__ == '__main__':
    prepare_data()
