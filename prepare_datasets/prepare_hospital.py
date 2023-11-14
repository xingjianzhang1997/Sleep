# -*- coding: utf-8 -*-
"""
1.Function：读取txt标签数据和EDF数据合并，数据清洗和数据存储。
2.Author：xingjian.zhang
3.Time：20231101
4.Others：This is a temporary script file.
"""

import pandas
import numpy as np
from datetime import datetime, timedelta
from mne.io import read_raw_edf

ann2label = {
    "清醒": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
    "未评分": 5
}


# 读取TXT数据，返回2维矩阵[3，N]
def readTXTFiles(file):
    data = []
    timeList = []
    epochList = []
    labelList = []

    # 清洗无用的标签信息行数
    lines = pandas.read_table(file, header=None)
    for i in range(len(lines)):
        curLine = lines[0][i]
        curList = curLine.split(',')
        curTime = curList[0]  # 类型是str
        curEpoch = curList[1]  # 类型是str
        curLabel = curList[2]  # 类型是str
        curLabel = ann2label[curLabel]

        # 判断list是否已增加有效数据
        if len(labelList) != 0:
            # 判断睡眠分期标记是否重复
            if curLabel == labelList[-1]:
                continue

        timeList.append(curTime)
        epochList.append(curEpoch)
        labelList.append(curLabel)

    timeArray = np.array(timeList)
    epochArray = np.array(epochList)
    labelArray = np.array(labelList)

    data = np.vstack((timeArray, epochArray, labelArray))
    return data


EPOCH_SEC_SIZE = 30
nameSubject = "ZengTianyi"  # ChengXiuYun
temp_flag = False  # 数据和标签没对齐再启动
select_ch_list = ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1"]  # E1-M2, E2-M2

edfPath = "/home/xingjian.zhang/sleep/0_data/02_hospitalRawdata/ZengTianyi.edf"
txtPath = "/home/xingjian.zhang/sleep/0_data/02_hospitalRawdata/ZengTianyi.txt"
savePath = "/home/xingjian.zhang/sleep/0_data/03_hospitalNPZdata/ZengTianyi_"
edfPath = edfPath.replace("ZengTianyi", nameSubject)
txtPath = txtPath.replace("ZengTianyi", nameSubject)
savePath = savePath.replace("ZengTianyi", nameSubject)

# 读取EDF数据
rawdata = read_raw_edf(edfPath, preload=True, verbose=False, stim_channel=None)
# 读取TXT数据
files = readTXTFiles(txtPath)
print("EDF和标签数据加载完毕")

# 检测raw和ann是不是同一个时间开始
raw_start_time = str(rawdata.info['meas_date'])
raw_start_time = raw_start_time.split(" ")[1]
raw_start_time = raw_start_time.split("+")[0]
txt_start_time = str(files[0, 0])

if temp_flag:
    print("数据时长修正，删除多余部分")
    delete_second = 6
else:
    delete_second = 0

if raw_start_time != txt_start_time and temp_flag is not True:
    raise Exception("文件报错, 标注时间和数据时间不一致，需人工check")
else:
    print("标注和记录时间已对齐")


for i in range(len(select_ch_list)):
    select_ch = select_ch_list[i]
    saveName = savePath + select_ch + ".npz"

    sampling_rate = rawdata.info['sfreq']
    raw_ch_df = rawdata.to_data_frame()[select_ch]
    raw_ch_df = raw_ch_df.to_frame()  # 将数据转换为一个新的 DataFrame，以确保它是一个独立的 DataFrame 对象
    raw_ch_df.set_index(np.arange(len(raw_ch_df)))  # 设置为一个新的整数索引，从0到 raw_ch_df 的长度减1

    # 该日期部分可以是任意日期，因为我们不关心日期，只是为了获取时间差
    fixed_date = "2023-01-01"

    # 最后1个label的数据直接删除，因为是未评分。
    labels = []  # 存储每个epoch的label
    label_idx = []  # 需保留的数据的索引值
    onsetSec = delete_second
    for j in range(len(files[0, :]) - 1):
        startTime = files[0, j]
        endTime = files[0, j + 1]
        curlabel = files[2, j]  # str

        # 将时间字符串解析为 datetime 对象，合并到固定日期
        startTime = datetime.strptime(fixed_date + " " + startTime, "%Y-%m-%d %H:%M:%S")
        endTime = datetime.strptime(fixed_date + " " + endTime, "%Y-%m-%d %H:%M:%S")

        # 如果结束时间在开始时间之前（跨越午夜），则增加一天的时间差
        if endTime < startTime:
            endTime += timedelta(days=1)

        # 计算时间差，转化为秒数
        timeDifference = endTime - startTime
        durationSec = timeDifference.total_seconds()
        durationEpoch = int(durationSec // EPOCH_SEC_SIZE)  # 取整
        deleteSec = int(durationSec % EPOCH_SEC_SIZE)  # 取余

        # 先跳过1个epoch都凑不齐的数据
        # if durationEpoch == 0:
        #     print(deleteSec)

        # 只保留需要数据的索引值
        if durationEpoch != 0 and curlabel != "5":
            usedSec = durationSec - deleteSec  # 去除最后不能整除的idx数据
            data_idx = int(onsetSec * sampling_rate) + np.arange(usedSec * sampling_rate, dtype=int)
            labelEpoch = np.ones(durationEpoch, dtype=int) * int(curlabel)
            labels.append(labelEpoch)
            label_idx.append(data_idx)

        onsetSec = onsetSec + durationSec

    labels = np.hstack(labels)
    label_idx = np.hstack(label_idx)
    raw_ch = raw_ch_df.values[label_idx] # 从原始数据中选择保留的indx对应的数值

    # 再次验证数据能被30-s整除 epochs
    if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
        raise Exception("原始数据不能被30S整除，有问题")

    n_epochs = int(len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate))
    x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
    y = labels.astype(np.int32)

    # 确保数据和标签是对应的
    assert len(x) == len(y)

    print(select_ch + " npz数据生成完毕")

    # Save
    save_dict = {
        "x": x,
        "y": y,
        "fs": sampling_rate,
        "ch_label": select_ch,
    }
    np.savez(saveName, **save_dict)

print("数据保存为npz格式完成，清洗后保存了 {}个epoch数据".format(len(y)))
print("W期有 {}个epoch数据".format(len(np.where(y==0)[0])))
print("N1期有 {}个epoch数据".format(len(np.where(y==1)[0])))
print("N2期有 {}个epoch数据".format(len(np.where(y==2)[0])))
print("N3期有 {}个epoch数据".format(len(np.where(y==3)[0])))
print("REM期有 {}个epoch数据".format(len(np.where(y==4)[0])))
