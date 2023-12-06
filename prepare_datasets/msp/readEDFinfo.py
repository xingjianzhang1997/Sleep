# -*- coding: utf-8 -*-
"""
1.Function：读取1例EDF数据的info信息
2.Author：xingjian.zhang
3.Time：20230925
4.Others：This is a temporary script file.
"""
import pandas as pd
from mne.io import read_raw_edf


def readAnnotFiles(path):
    dataTable = pd.read_table(path, header=None)
    dataTable = dataTable.drop([1, 2, 5], axis=1)  # 删除多余的列（已确认是无用信息行）
    keepClassNames = ['W', 'R', 'N1', 'N2', 'N3']
    condition = dataTable[0].isin(keepClassNames)
    filteredDataTable = dataTable[condition]  # 只保留符合条件的行

    return filteredDataTable


raw_path = "/data/xingjain.zhang/sleep/0_rawdata/msp/msp-S091.edf"
ann_path = "/data/xingjain.zhang/sleep/0_rawdata/msp/msp-S091.annot"


raw = read_raw_edf(raw_path, preload=True, verbose=False, stim_channel=None)
raw.plot()
annTable = readAnnotFiles(ann_path)

print(raw.info)
print(raw.info.ch_names)


print("完成")
