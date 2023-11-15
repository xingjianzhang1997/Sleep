"""
1.Function：生成训练数据
2.Author：xingjian.zhang
3.Time：20231115
4.Others：在原有基础上，从单通道变为两通道的数据。
"""

import glob
import math
import ntpath
import os
import shutil
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mne import read_annotations
from mne.io import read_raw_edf

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


def main():
    # 已经提取check过数据标签，标签一定在下列范围内
    ann2label = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4,
        "Sleep stage ?": 5,
        "Movement time": 5
    }

    stage_dict = {
        "W": 0,
        "N1": 1,
        "N2": 2,
        "N3": 3,
        "REM": 4,
        "UNKNOWN": 5
    }

    EPOCH_SEC_SIZE = 30  # 30S 为一个epoch

    # 设置数据保存的
    dataPath = '/home/xingjian.zhang/sleep/0_data/00_rawdata/sleep-telemetry/'
    savePath = '/home/xingjian.zhang/sleep/0_data/01_npzdata/03_ST_EOG/'
    select_ch = "EEG Fpz-Cz"  # EEG Fpz-Cz or EEG Pz-Oz

    num_W_epoch = 0
    num_N1_epoch = 0
    num_N2_epoch = 0
    num_N3_epoch = 0
    num_REM_epoch = 0

    # Output dir
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    else:
        shutil.rmtree(savePath)  # 用于递归删除指定目录以及其包含的所有文件和子目录
        os.makedirs(savePath)

    # Read raw and annotation EDF files
    psg_fnames = glob.glob(os.path.join(dataPath, "*PSG.edf"))  # 使用glob模块查找所有以 "*PSG.edf" 结尾的文件
    ann_fnames = glob.glob(os.path.join(dataPath, "*Hypnogram.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)  # 把列表转化为了numpy数组。
    ann_fnames = np.asarray(ann_fnames)

    for i in range(len(psg_fnames)):
        print("正在处理的文件编号： {}".format(i))
        raw = read_raw_edf(psg_fnames[i], preload=True, verbose=False, stim_channel=None)
        sampling_rate = raw.info['sfreq']
        raw_ch_df = raw.to_data_frame()[select_ch]
        raw_ch_df = raw_ch_df.to_frame()  # 将数据转换为一个新的 DataFrame，以确保它是一个独立的 DataFrame 对象
        raw_ch_df.set_index(np.arange(len(raw_ch_df)))  # 设置为一个新的整数索引，从0到 raw_ch_df 的长度减1

        # 只是为了获取ann的header信息，正确读取ann的函数是read_annotations
        warnings.filterwarnings("ignore", category=RuntimeWarning)  # 忽略 RuntimeWarning
        ann = read_raw_edf(ann_fnames[i], verbose=False)
        warnings.resetwarnings()  # 恢复警告设置
        raw_start_time = raw.info['meas_date'].strftime("%Y-%m-%d %H:%M:%S UTC")
        ann_start_time = ann.info['meas_date'].strftime("%Y-%m-%d %H:%M:%S UTC")

        # 检测一下raw和ann是不是同一个时间开始
        if raw_start_time != ann_start_time:
            print("报错的文件编号： {}, 时间不对齐".format(i))
            continue

        ann = read_annotations(ann_fnames[i])

        # Generate label and remove indices
        remove_idx = []  # indicies of the data that will be removed
        labels = []  # indicies of the data that have labels
        label_idx = []
        for j in range(len(ann.description)):
            onset_sec = ann.onset[j]
            duration_sec = ann.duration[j]
            ann_str = ann.description[j]
            label = ann2label[ann_str]

            # 排除数据
            if label != 5:
                # 检测标签是不是30S的整数倍。
                if duration_sec % EPOCH_SEC_SIZE != 0:
                    raise Exception("Something wrong")

                # 划分为30S的epoch并记录时间索引点。
                duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
                label_epoch = np.ones(duration_epoch, dtype=int) * label
                labels.append(label_epoch)
                idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=int)
                label_idx.append(idx)
                # print ("Include onset:{}, duration:{}, label:{} ({})".format(
                #     onset_sec, duration_sec, label, ann_str))

            else:
                idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=int)
                remove_idx.append(idx)
                # print ("Remove onset:{}, duration:{}, label:{} ({})".format(
                #         onset_sec, duration_sec, label, ann_str))

        labels = np.hstack(labels)
        print("before remove unwanted: {}".format(np.arange(len(raw_ch_df)).shape))
        if len(remove_idx) > 0:
            remove_idx = np.hstack(remove_idx)  # remove_idx 是一个列表，其中每个元素是一个索引数组，需要将它们合并成一个单一的数组。
            select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)  # np.setdiff1d 函数：这个函数用于计算两个数组之间的差集。
        else:
            select_idx = np.arange(len(raw_ch_df))
        print("after remove unwanted: {}".format(select_idx.shape))

        # 删除未打标签的数据
        print("before intersect label: {}".format(select_idx.shape))
        label_idx = np.hstack(label_idx)
        select_idx = np.intersect1d(select_idx, label_idx)  # 用于计算两个数组的交集
        print("after intersect label: {}".format(select_idx.shape))

        # 删除额外的标签的数据（经常会出现最后一个标签对应的数据不全的问题）
        if len(label_idx) > len(select_idx):
            print("before remove extra labels: {}, {}".format(select_idx.shape, labels.shape))
            # 删除最后的标签和不全的数据
            n_trims = len(select_idx) % int(EPOCH_SEC_SIZE * sampling_rate)
            n_label_trims = int(math.ceil(n_trims / (EPOCH_SEC_SIZE * sampling_rate)))
            if n_trims == 0:
                print("注意！标签的数据远远多于rawdata，估计有问题，跳过！")
                continue
            if n_label_trims != 1:
                print("注意！删除的标签数据超过1，需要check！")
                print("n_trims: {}".format(n_trims))
                print("n_label_trims: {}".format(n_label_trims))
            # 删除列表中从倒数第n_trims个元素开始到最后一个元素的所有元素
            select_idx = select_idx[:-n_trims]
            labels = labels[:-n_label_trims]
            print("after remove extra labels: {}, {}".format(select_idx.shape, labels.shape))

            # Remove movement and unknown stages if any
        raw_ch = raw_ch_df.values[select_idx]

        # Verify that we can split into 30-s epochs
        if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
            raise Exception("原始数据不能被30S整除，有问题")

        n_epochs = int(len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate))

        # Get epochs and their corresponding labels
        x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
        y = labels.astype(np.int32)
        # print("数据切割为epochs完成，共有 {}个epoch".format(n_epochs))

        # 确保数据和标签是对应的
        assert len(x) == len(y)

        # 避免wake数据太多，最前后的wake期数据进行截取，最多选择120个epoch数据
        # w_edge_mins = 60
        # nw_idx = np.where(y != stage_dict["W"])[0] # 获取非清醒状态下的数据epoch单位
        # start_idx = nw_idx[0] - (w_edge_mins)
        # end_idx = nw_idx[-1] + (w_edge_mins)
        # if start_idx < 0: start_idx = 0
        # if end_idx >= len(y): end_idx = len(y) - 1
        # select_idx = np.arange(start_idx, end_idx+1)
        # print("Data before selection: {}, {}".format(x.shape, y.shape))
        # x = x[select_idx]
        # y = y[select_idx]
        # print("Data after selection: {}, {}".format(x.shape, y.shape))

        # Save
        filename = ntpath.basename(psg_fnames[i]).replace("-PSG.edf", ".npz")
        save_dict = {
            "x": x,
            "y": y,
            "fs": sampling_rate,
            "ch_label": select_ch,
        }
        np.savez(os.path.join(savePath, filename), **save_dict)
        stage_dict = {
            "W": 0,
            "N1": 1,
            "N2": 2,
            "N3": 3,
            "REM": 4,
            "UNKNOWN": 5
        }
        print("数据保存为npz格式完成，选择保存了 {}个epoch数据\n".format(len(y)))
        num_W_epoch = num_W_epoch + len(np.where(y == stage_dict["W"])[0])
        num_N1_epoch = num_N1_epoch + len(np.where(y == stage_dict["N1"])[0])
        num_N2_epoch = num_N2_epoch + len(np.where(y == stage_dict["N2"])[0])
        num_N3_epoch = num_N3_epoch + len(np.where(y == stage_dict["N3"])[0])
        num_REM_epoch = num_REM_epoch + len(np.where(y == stage_dict["REM"])[0])
        all_epoch = num_W_epoch + num_N1_epoch + num_N2_epoch + num_N3_epoch + num_REM_epoch

    print("=================================")
    print("W共有 {} 个epoch数据".format(num_W_epoch))
    print("N1共有 {} 个epoch数据".format(num_N1_epoch))
    print("N2共有 {} 个epoch数据".format(num_N2_epoch))
    print("N3共有 {} 个epoch数据".format(num_N3_epoch))
    print("REM共有 {} 个epoch数据".format(num_REM_epoch))
    print("共有 {} 个epoch数据".format(all_epoch))

    sizes = [num_W_epoch, num_N1_epoch, num_N2_epoch, num_N3_epoch, num_REM_epoch]
    labels = ['stage_W', 'stage_N1', 'stage_N2', 'stage_N3/4', 'stage_REM']
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']
    plt.pie(sizes, labels=labels, colors=colors, autopct=make_autopct(sizes))
    plt.title("Sleep-EDF SC set")
    plt.show()


if __name__ == "__main__":
    main()
