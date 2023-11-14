# -*- coding: utf-8 -*-
"""
1.Function：读取Apples数据库的annot文件的标签数据
2.Author：xingjian.zhang
3.Time：20231106
4.Others：This is a temporary script file.
"""

import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from datetime import datetime, timedelta

# plt.rcParams是一个全局配置对象,设置显示中文字体和负号
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (10, 8)


# 用来让最后的饼图同时显示比例和具体数值
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        # 同时显示数值和占比的饼图
        return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)

    return my_autopct


annotPath = "/home/xingjian.zhang/sleep/0_data/04_applesRawdata/"
annotfnames = glob.glob(os.path.join(annotPath, "*.annot"))
annotfnames.sort()

Wake = 0
REM = 0
N1 = 0
N2 = 0
N3 = 0
annotSecond = 0
tempF = 0

for i in range(len(annotfnames)):
    # 读取数据，形成table方便后续处理。
    dataTable = pd.read_table(annotfnames[i], header=None)

    # 删除多余的列
    dataTable = dataTable.drop([1, 2, 5], axis=1)

    # 只保留符合条件的行
    keepClassNames = ['W', 'R', 'N1', 'N2', 'N3']
    condition = dataTable[0].isin(keepClassNames)
    filteredDataTable = dataTable[condition]
    print("检测数据： {}".format(annotfnames[i]))
    for j in range(len(filteredDataTable)-1):
        if filteredDataTable.iloc[j, 2] != filteredDataTable.iloc[j+1, 1]:
            print("数据不连续！！！")

    # 使用 value_counts 统计不同类别的行数
    class_counts = filteredDataTable[0].value_counts()

    # 计算不同类别的行数
    Wake = Wake + class_counts.get('W', 0)
    REM = REM + class_counts.get('R', 0)
    N1 = N1 + class_counts.get('N1', 0)
    N2 = N2 + class_counts.get('N2', 0)
    N3 = N3 + class_counts.get('N3', 0)


all_epoch = Wake + REM + N1 + N2 + N3

# 判断是否有未打标签的片段

print("=================================")
print("W共有 {} 个epoch数据".format(Wake))
print("N1共有 {} 个epoch数据".format(N1))
print("N2共有 {} 个epoch数据".format(N2))
print("N3共有 {} 个epoch数据".format(N3))
print("REM共有 {} 个epoch数据".format(REM))
print("共有 {} 个epoch数据".format(all_epoch))

sizes = [Wake, N1, N2, N3, REM]
labels = ['stage_W', 'stage_N1', 'stage_N2', 'stage_N3', 'stage_REM']
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']
plt.pie(sizes, labels=labels, colors=colors, autopct=make_autopct(sizes))
plt.title("Apples set")
plt.show()

