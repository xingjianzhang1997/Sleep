"""
1.Function：绘制睡眠趋势图
2.Author：xingjian.zhang
3.Time：20231201
4.Others：1）增加规则修改
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta


def plotTime(data, flag_modi=False):
    """
    模型的输出结果绘制成睡眠趋势变化图
    data：模型的预测结果标签值
    flag_modi：是否要加入规则修改data
    """
    time_interval = 30  # 每个标签代表的时间间隔
    sleep_stages = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}  # 映射数值到对应的睡眠阶段
    if flag_modi:
        flag_modi_1 = False
        flag_modi_2 = True
        flag_modi_3 = True
        flag_modi_4 = True
        flag_modi_5 = True
        if flag_modi_1:
            "数据开始记录的前20分钟都记录为wake"
            time_wake = 20  # 单位为分钟
            num_label = int(time_wake * 60 / time_interval)
            data[0:num_label] = 0
        if flag_modi_3:
            "至少有连续2个epoch被判断为REM期，才可以判定为REM期"
            min_rem_epochs = 2
            for i in range(1, len(data) - min_rem_epochs + 1):
                if all(data[i + j] == 4 for j in range(min_rem_epochs)):
                    data[i:i + min_rem_epochs] = 4
        if flag_modi_4:
            "如果2个REM epoch之间间隔不超过5个Non-REM epoch，那么这些epoch也将被评定为REM"
            max_non_rem_gap = 5
            for i in range(1, len(data) - 1):
                if data[i] == 4 and data[i - 1] == 4 and data[i + 1] != 4:
                    gap_count = 0
                    j = i + 1
                    while j < len(data) and data[j] != 4:
                        gap_count += 1
                        j += 1
                    if gap_count <= max_non_rem_gap:
                        data[i + 1:j] = 4
        if flag_modi_5:
            "如果当前epoch被判定为觉醒，后面连续的4个epoch也将被判定为觉醒"
            for i in range(1, len(data) - 3):
                if data[i] == 0 and all(data[i + j] == 0 for j in range(1, 5)):
                    data[i:i + 4] = 0
        if flag_modi_2:
            "数值变化至少保持2个epoch才可以变化"
            min_epochs_to_change = 2
            current_epoch_count = 0
            for i in range(1, len(data)):
                if data[i] != data[i - 1]:
                    current_epoch_count += 1
                    if current_epoch_count < min_epochs_to_change:
                        data[i] = data[i - 1]
                else:
                    current_epoch_count = 0

    # 起始时间（晚上9点）
    start_time = datetime.strptime("21:00", "%H:%M")

    # 生成时间轴
    time_axis = [start_time + timedelta(seconds=i * time_interval) for i in range(len(data))]

    # 绘制横线图形和连接的竖线
    for i in range(len(data) - 1):
        t1, t2 = time_axis[i], time_axis[i + 1]
        value1, value2 = data[i], data[i + 1]

        # 绘制横线
        plt.hlines(value1, t1, t2, colors='blue')

        # 如果数值发生变化，绘制连接的竖线
        if value1 != value2:
            plt.vlines(t2, min(value1, value2), max(value1, value2))

    # 绘制最后一个数据点对应的横线
    plt.hlines(data[-1], time_axis[-1], time_axis[-1] + timedelta(seconds=time_interval), colors='blue')

    plt.xlabel('Time')
    plt.ylabel('Sleep Stage')
    plt.title('MDSK Sleep Stage over Time')

    # 修改刻度标签
    plt.xlim(datetime.strptime("21:00", "%H:%M"), datetime.strptime("06:00", "%H:%M") + timedelta(days=1))
    plt.yticks(list(sleep_stages.keys()), list(sleep_stages.values()))
    plt.gca().invert_yaxis()  # 倒置Y轴

    # 格式化x轴的时间显示
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.gcf().autofmt_xdate()

    plt.show()


def main():
    dataPath = "/home/xingjian.zhang/sleep/3_independent_result/MDSK_ST_FPZ-Cz&EOG_model/y_pre.npy"  # AI测试的结果
    y_pre = np.load(dataPath)
    plotTime(y_pre, flag_modi=True)

if __name__ == "__main__":
    main()
    