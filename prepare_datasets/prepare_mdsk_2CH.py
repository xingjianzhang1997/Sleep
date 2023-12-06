"""
1.Function：生成训练数据
2.Author：xingjian.zhang
3.Time：20231130
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


def main():

    EPOCH_SEC_SIZE = 30  # 30S 为一个epoch

    # 设置数据保存的
    dataPath = '/home/xingjian.zhang/sleep/0_data/98_mdskRawdata'
    savePath = '/home/xingjian.zhang/sleep/0_data/99_mdskNPZdata'
    select_ch = ["eeg", "eog"]

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    else:
        shutil.rmtree(savePath)  # 用于递归删除指定目录以及其包含的所有文件和子目录
        os.makedirs(savePath)

    # Read raw and annotation EDF files
    psg_fnames = glob.glob(os.path.join(dataPath, "*.edf"))  # 使用glob模块查找所有以 "*.edf" 结尾的文件
    psg_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)  # 把列表转化为了numpy数组。

    for i in range(len(psg_fnames)):
        print("正在处理的文件编号： {}".format(i))
        raw = read_raw_edf(psg_fnames[i], preload=True, verbose=False, stim_channel=None)
        sampling_rate = raw.info['sfreq']
        raw_ch_df = raw.to_data_frame()[select_ch]
        n_trims = len(raw_ch_df) % int(EPOCH_SEC_SIZE * sampling_rate)
        raw_ch = raw_ch_df.values[:-n_trims]

        # Verify that we can split into 30-s epochs
        if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
            raise Exception("原始数据不能被30S整除，有问题")

        n_epochs = int(len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate))

        # Get epochs and their corresponding labels
        x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
        y = np.ones(n_epochs).astype(np.int32)
        print("数据切割为epochs完成，共有 {}个epoch".format(n_epochs))

        # Save
        filename = ntpath.basename(psg_fnames[i]).replace(".edf", ".npz")
        save_dict = {
            "x": x,
            "y": y,
            "fs": sampling_rate,
            "start_time": start_time,
            "end_time": end_time,
        }
        np.savez(os.path.join(savePath, filename), **save_dict)


if __name__ == "__main__":
    main()
