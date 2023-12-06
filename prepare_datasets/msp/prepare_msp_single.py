"""
1.Function：读取annot标签数据和EDF数据合并，数据清洗和数据存储.
2.Author：xingjian.zhang
3.Time：20231120
4.Others：在原基础上，同时生成不同通道的NPZ数据.
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from mne.io import read_raw_edf
import matplotlib.pyplot as plt

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


def readAnnotFiles(path):
    dataTable = pd.read_table(path, header=None)
    dataTable = dataTable.drop([1, 2, 5], axis=1)  # 删除多余的列（已确认是无用信息行）
    keepClassNames = ['W', 'R', 'N1', 'N2', 'N3']
    condition = dataTable[0].isin(keepClassNames)
    filteredDataTable = dataTable[condition]  # 只保留符合条件的行

    return filteredDataTable


def checkTime(time_edf, time_ann):
    """判断标签和数据的时间是否一致，不一致调整数据的开始时间"""
    onsetSec = 0
    flag_del = False
    # 该日期部分可以是任意日期，因为我们不关心日期，只是为了获取时间差
    fixed_date = "2023-01-01"
    # 将时间字符串解析为 datetime 对象，合并到固定日期
    time_edf = datetime.strptime(fixed_date + " " + time_edf, "%Y-%m-%d %H:%M:%S")
    time_ann = datetime.strptime(fixed_date + " " + time_ann, "%Y-%m-%d %H:%M:%S")

    # 创建12点的 datetime 对象
    thresholdTime = datetime(time_edf.year, time_edf.month, time_edf.day, 12, 0, 0)

    # 可能会从12点之后才开始计算时间, 判断是否过夜了
    if time_edf < thresholdTime:
        time_edf += timedelta(days=1)
    if time_ann < thresholdTime:
        time_ann += timedelta(days=1)

    if time_edf == time_ann:
        print("EDF和标签时间一致，不需要额外处理")
    elif time_edf < time_ann:
        timeDifference = time_ann - time_edf
        differentSecond = timeDifference.total_seconds()
        onsetSec = differentSecond  # 使用EDF的开始时间
        print("EDF比标签开始的时间早")
        print(time_edf)
        print(time_ann)
    elif time_edf > time_ann:
        flag_del = True
        print("EDF比标签开始的时间晚，这种情况很不寻常，估计数据有问题")
        print(time_edf)
        print(time_ann)

    return onsetSec, flag_del


def cleanData(rawdata, labels, data_idx, select_ch, sampling_rate):
    raw_ch_df = rawdata.to_data_frame()[select_ch]
    raw_ch_df = raw_ch_df.to_frame()  # 将数据转换为一个新的 DataFrame，以确保它是一个独立的 DataFrame 对象.
    raw_ch_df.set_index(np.arange(len(raw_ch_df)))  # 设置为一个新的整数索引.

    # 可能存在结尾EDF数据比标签数据短的情况（数据损坏导致的？）
    if data_idx[-1] > len(raw_ch_df) - 1:
        deleteIndx = data_idx[-1] - (len(raw_ch_df) - 1)
        deleteIndxEpoch = int(deleteIndx // (EPOCH_SEC_SIZE * sampling_rate))  # 取整
        deleteIndxEpoch_remain = int(deleteIndx % (EPOCH_SEC_SIZE * sampling_rate))  # 取余

        if deleteIndxEpoch_remain == 0:
            labels = labels[:-deleteIndxEpoch]
            data_idx = data_idx[:-deleteIndx]
        else:
            deleteIndxEpoch = deleteIndxEpoch + 1
            labels = labels[:-deleteIndxEpoch]
            deleteIndxRaw = deleteIndx + int(EPOCH_SEC_SIZE * sampling_rate - deleteIndxEpoch_remain)
            data_idx = data_idx[:-deleteIndxRaw]
        print("EDF数据比标签数据短, 删除最后{}个epoch".format(deleteIndxEpoch))

    raw_ch = raw_ch_df.values[data_idx]  # 从原始数据中选择保留的indx对应的数值

    # 再次验证数据能被30-s整除 epochs
    if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
        raise Exception("原始数据不能被30S整除，有问题")

    n_epochs = int(len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate))
    x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
    y = labels.astype(np.int32)

    # 确保数据和标签是对应的
    assert len(x) == len(y)

    return x, y


def combineEDFandAnnot(path):
    ann2label = {
        "W": 0,
        "N1": 1,
        "N2": 2,
        "N3": 3,
        "R": 4,
    }

    num_W_epoch = 0
    num_N1_epoch = 0
    num_N2_epoch = 0
    num_N3_epoch = 0
    num_REM_epoch = 0

    psgFiles = glob.glob(os.path.join(path, "*.edf"))
    annotFiles = glob.glob(os.path.join(path, "*.annot"))
    psgFiles.sort()
    annotFiles.sort()
    numFiles = len(psgFiles)
    print("PSG文件例数： {}".format(numFiles))
    list_otherHZ = []

    for i in range(numFiles):

        print("*****正在处理的文件编号：{} *****".format(i))
        print("正在处理的文件地址：{}".format(psgFiles[i]))
        print("正在处理的文件地址：{}".format(annotFiles[i]))

        rawdata = read_raw_edf(psgFiles[i], preload=True, verbose=False, stim_channel=None)
        annotdata = readAnnotFiles(annotFiles[i])  # 类型是frame

        sampling_rate = int(rawdata.info['sfreq'])
        # 跳过采样率不为500Hz
        if sampling_rate != 500:
            list_otherHZ.append(i)
            continue

        raw_startTime = str(rawdata.info['meas_date']).split(" ")[1]
        raw_startTime = raw_startTime.split("+")[0]
        ann_startTime = annotdata.iloc[0, 1]

        # 检查数据和标签的时间是否对齐, 不对齐则矫正
        onsetSec, flag_del = checkTime(raw_startTime, ann_startTime)
        if flag_del:
            continue

        durationSecond = len(annotdata) * 30
        labels = annotdata.iloc[:, 0].to_numpy()
        mapping_function = np.frompyfunc(ann2label.get, 1, 1)  # 将数组元素映射为对应的值
        labels = mapping_function(labels)

        data_idx = int(onsetSec * sampling_rate) + np.arange(durationSecond * sampling_rate, dtype=int)

        for j in range(len(select_ch_list)):
            select_ch = select_ch_list[j]

            savePath_ch = savePath + "/" + select_ch
            if not os.path.exists(savePath_ch):
                os.makedirs(savePath_ch)

            data_X, lable_y = cleanData(rawdata, labels, data_idx, select_ch, sampling_rate)

            filename = psgFiles[i].split("/")[-1].replace(".edf", ".npz")
            save_dict = {
                "x": data_X,
                "y": lable_y,
                "fs": sampling_rate,
                "ch_label": select_ch,
            }
            np.savez(os.path.join(savePath_ch, filename), **save_dict)

        num_W_epoch = num_W_epoch + len(np.where(lable_y == 0)[0])
        num_N1_epoch = num_N1_epoch + len(np.where(lable_y == 1)[0])
        num_N2_epoch = num_N2_epoch + len(np.where(lable_y == 2)[0])
        num_N3_epoch = num_N3_epoch + len(np.where(lable_y == 3)[0])
        num_REM_epoch = num_REM_epoch + len(np.where(lable_y == 4)[0])
        print("保存了{}个epoch数据为NPZ格式".format(len(lable_y)))

    print("Other Hz {}".format(list_otherHZ))
    print("**************************************************")
    all_epoch = num_W_epoch + num_N1_epoch + num_N2_epoch + num_N3_epoch + num_REM_epoch
    print("数据保存为npz格式完成，清洗后保存了 {}个epoch数据".format(all_epoch))
    print("W期有 {}个epoch数据".format(num_W_epoch))
    print("N1期有 {}个epoch数据".format(num_N1_epoch))
    print("N2期有 {}个epoch数据".format(num_N2_epoch))
    print("N3期有 {}个epoch数据".format(num_N3_epoch))
    print("REM期有 {}个epoch数据".format(num_REM_epoch))

    sizes = [num_W_epoch, num_N1_epoch, num_N2_epoch, num_N3_epoch, num_REM_epoch]
    labels = ['stage_W', 'stage_N1', 'stage_N2', 'stage_N3', 'stage_REM']
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']
    plt.pie(sizes, labels=labels, colors=colors, autopct=make_autopct(sizes))
    plt.title("Apples set")
    plt.show()

if __name__ == "__main__":
    select_ch_list = ["EMG", "ECG"]  # ECG, C3_M2, C4_M1, O1_M2, O2_M1
    filePath = "/data/xingjain.zhang/sleep/0_rawdata/msp"
    savePath = "/data/xingjain.zhang/sleep/1_npzdata/msp"

    EPOCH_SEC_SIZE = 30



