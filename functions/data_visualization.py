# -*- coding: utf-8 -*-
"""
1.Function：对Sleep-EDF进行数据处理
2.Author：xingjian.zhang
3.Time：20230912-201915
4.Others：V1版本
"""
import mne
import glob
import os
import warnings
import numpy as np
from mne import read_annotations
from mne.io import concatenate_raws, read_raw_edf

import matplotlib.pyplot as plt

# plt.rcParams是一个全局配置对象,设置显示中文字体和负号
plt.rcParams['font.family'] = 'DejaVu Sans' 
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"]=(10, 8)  

# 用来让最后的饼图同时显示比例和具体数值
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        # 同时显示数值和占比的饼图
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct


# 把数据地址读到数组里。
file_path = r'C:\ZXJ\python\DL\1_Sleep\0_data\sleep-cassette' 
psg_fnames = glob.glob(os.path.join(file_path, '*PSG.edf'))  # 匹配指定目录中的文件。
ann_fnames = glob.glob(os.path.join(file_path, '*Hypnogram.edf'))
psg_fnames.sort()
ann_fnames.sort()
psg_fnames = np.asarray(psg_fnames) # 将输入数据转换为 NumPy 数组
ann_fnames = np.asarray(ann_fnames)

# 设置通道类型，将通道名与类型进行映射
mapping = {'EEG Fpz-Cz': 'eeg',
           'EEG Pz-Oz': 'eeg',
           'EOG horizontal': 'misc',
           'EMG submental': 'misc'}
scalings = {'eeg': 2*1e-4}
channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']

# 已经提前通过checkLables函数读取了sleep-EDF的所有数据(ST和SC），确保标签只有如下列表。
# 原始标签转化为数字标签
annotation2event_id = {'Sleep stage W': 1,
                        'Sleep stage 1': 2,
                        'Sleep stage 2': 3,
                        'Sleep stage 3': 4,
                        'Sleep stage 4': 4,
                        'Sleep stage R': 5,
                        'Movement time': 6,
                        'Sleep stage ?': 7}


# 根据AASM规则，N3和N4可以视为1类，只分这5类数据
event_id = {'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3/4': 4,
            'Sleep stage R': 5}


i = 0
print("当前处理的编号: %d" %i)
raw_data = read_raw_edf(psg_fnames[i], preload=True, verbose=False) # preload=True意味数据一次加载到内存里，verbose控制信息是否弹出。
annot_data = read_annotations(ann_fnames[i])
raw_data.set_annotations(annot_data, emit_warning=False) 
warnings.filterwarnings("ignore", category=RuntimeWarning) # 忽略 RuntimeWarning
raw_data.set_channel_types(mapping)
warnings.resetwarnings() # 恢复警告设置
raw_data.pick(channels)  # 选择感兴趣的通道
# raw_data.plot(start=60, duration=300, scalings=scalings)

events_data, _ = mne.events_from_annotations(
    raw_data, event_id=annotation2event_id, chunk_duration=30, verbose=False)

# fig = mne.viz.plot_events(events_data, event_id=event_id,
#                           sfreq=raw_data.info['sfreq'],
#                           first_samp=events_data[0, 0])

# 把每一个event数据转化为epoch（睡眠分期中的最小单位（30S））
# mne.Epochs将数据划分为不同的时间窗口，以便进行事件相关分析和统计。
# 如果不减去一个采样点的时间，那么tmax指定的时间窗口将包括30秒以及最后一个采样点数据
tmax = 30. - 1. / raw_data.info['sfreq']
epochs_all_data = mne.Epochs(raw_data, events_data, tmin=0, tmax=tmax, event_id=event_id,
                    preload=True,baseline=None, verbose=False)


# for i in range(1, len(psg_fnames)):
for i in range(70, len(psg_fnames)):
    if i == 40 or i == 63 or 65 <= i <= 68 or i == 109 or 123<=i<=124 or 134<=i<=142 or i==144:
        # SC数据集：i=40/63/65/66/67/68/109/123/124/134/135/136/137/138/139/140/141/142/144没有N3和N4。
        # ST数据集：i=41没有N3和N4，导致标签类别不一样，mne.concatenate_epochs会报错。
        continue
    
    print("当前处理的编号: %d" %i)
    raw_data = read_raw_edf(psg_fnames[i], preload=True, verbose=False) # preload=True意味数据一次加载到内存里，verbose控制信息是否弹出。
    annot_data = read_annotations(ann_fnames[i])
    raw_data.set_annotations(annot_data, emit_warning=False) 
    warnings.filterwarnings("ignore", category=RuntimeWarning) # 忽略 RuntimeWarning
    raw_data.set_channel_types(mapping)
    warnings.resetwarnings() # 恢复警告设置
    raw_data.pick(channels)  # 选择感兴趣的通道
    
    events_data, _ = mne.events_from_annotations(
        raw_data, event_id=annotation2event_id, chunk_duration=30, verbose=False)
    
    tmax = 30. - 1. / raw_data.info['sfreq']
    epochs_data = mne.Epochs(raw_data, events_data, tmin=0, tmax=tmax, event_id=event_id,
                        preload=True,baseline=None, verbose=False)
    epochs_all_data = mne.concatenate_epochs([epochs_all_data, epochs_data], verbose=False)
    
    # 对于SC也没有足够的内存一次都加载进去。
    # No baseline correction applied
    # RuntimeWarning: Concatenation of Annotations within Epochs is not supported yet. All annotations will be dropped.

stage_W = len(epochs_all_data['Sleep stage W'])
stage_1 = len(epochs_all_data['Sleep stage 1'])
stage_2 = len(epochs_all_data['Sleep stage 2'])
stage_3 = len(epochs_all_data['Sleep stage 3/4'])
stage_R = len(epochs_all_data['Sleep stage R'])
print(epochs_all_data)


# SC 0
# <Epochs |  2650 events (all good), 0 – 29.99 s, baseline off, ~121.3 MB, data loaded,
#  'Sleep stage W': 1997
#  'Sleep stage 1': 58
#  'Sleep stage 2': 250
#  'Sleep stage 3/4': 220
#  'Sleep stage R': 125>

# SC  0-69（含0）
# <EpochsArray |  174532 events (all good), 0 – 29.99 s, baseline off, ~7.80 GB, data loaded,
#  'Sleep stage W': 119992
#  'Sleep stage 1': 5788
#  'Sleep stage 2': 28992
#  'Sleep stage 3/4': 7716
#  'Sleep stage R': 12044>
 
# SC  71-153（含0）


# 5 个数值
sizes = [stage_W, stage_1, stage_2, stage_3, stage_R]
# 对应的标签
labels = ['stage_W', 'stage_1', 'stage_2', 'stage_3/4', 'stage_R']
# 饼图颜色
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']
# 创建饼图
plt.pie(sizes, labels=labels, colors=colors, autopct=make_autopct(sizes))
# 饼图标题
plt.title("Sleep-EDF ST set")
# 显示饼图
plt.show()

