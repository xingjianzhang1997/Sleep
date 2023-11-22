# -*- coding: utf-8 -*-
"""
1.Function：读取1例EDF数据的info信息
2.Author：xingjian.zhang
3.Time：20230925
4.Others：This is a temporary script file.
"""

from mne.io import read_raw_edf

path = "/home/xingjian.zhang/sleep/0_data/04_applesRawdata/apples-570317.edf"


raw = read_raw_edf(path, preload=True, verbose=False, stim_channel=None)
raw.plot()


print(raw.info)
print(raw.info.ch_names)


# annot_data = read_annotations(ann_fnames[i])
# raw_data.set_annotations(annot_data, emit_warning=False)
