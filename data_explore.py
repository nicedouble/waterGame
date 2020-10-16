#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time     : 2020/8/28 9:35
@Author   : ji hao ran
@File     : data_explore.py
@Project  : waterGame
@Software : PyCharm
"""

import seaborn as sns
from utils import *

# 数据
train_set, valid_set1, valid_set2, valid_set3 = data_preprocessing()

# 图表输出主路径
output_path = 'E:/waterGame/explore/'

# 描述统计
train_set.describe().to_csv(output_path + 'describe/train_set.csv')
valid_set1.describe().to_csv(output_path + 'describe/valid_set1.csv')
valid_set2.describe().to_csv(output_path + 'describe/valid_set2.csv')
valid_set3.describe().to_csv(output_path + 'describe/valid_set3.csv')

# 时序图
for year in range(2013, 2018):
    for var in ['Qi', 'T', 'w', 'Rsum', 'D1', 'D2', 'D3', 'D4', 'D5']:
        plt.switch_backend('agg')
        plt.figure(figsize=(16, 8))
        title = '_'.join(['train_set', str(year), var])
        plt.plot(train_set.loc[train_set.index.year == year, var], label=var)
        plt.legend()
        plt.title(title)
        plt.savefig(output_path + 'plot/train_set/' + title + '.png')
        plt.close()

for data, name in zip([valid_set1, valid_set2, valid_set3], ['valid_set1', 'valid_set2', 'valid_set3']):
    for var in ['Qi', 'T', 'w', 'Rsum', 'D1', 'D2', 'D3', 'D4', 'D5']:
        plt.switch_backend('agg')
        plt.figure(figsize=(16, 8))
        title = '_'.join([name, var])
        plt.plot(data.loc[:, var], label=var)
        plt.legend()
        plt.title(title)
        plt.savefig(output_path + 'plot/' + name + '/' + title + '.png')
        plt.close()

# 盒形图

# 直方图

# 散点图矩阵
# sns.pairplot(train_set, hue='rainy_season', markers=["o", "s"])
# sns.pairplot(valid_set1, hue='rainy_season', markers=["o", "s"])
# sns.pairplot(valid_set2, hue='rainy_season', markers=["o", "s"])
# sns.pairplot(valid_set3, hue='rainy_season', markers=["o", "s"])

# 相关性矩阵图
# 相关性矩阵
train_set_corr = train_set.corr()
train_set_corr.to_csv(output_path + 'corr/train_set.csv')
valid_set1_corr = valid_set1.corr()
valid_set1_corr.to_csv(output_path + 'corr/valid1_set.csv')
valid_set2_corr = valid_set2.corr()
valid_set2_corr.to_csv(output_path + 'corr/valid2_set.csv')
valid_set3_corr = valid_set3.corr()
valid_set3_corr.to_csv(output_path + 'corr/valid3_set.csv')
# 绘图
sns.heatmap(train_set_corr, annot=True)
sns.heatmap(valid_set1_corr, annot=True)
sns.heatmap(valid_set2_corr, annot=True)
sns.heatmap(valid_set3_corr, annot=True)
