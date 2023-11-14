# -*- coding: utf-8 -*-
"""
1.Function：
2.Author：xingjian.zhang
3.Time：20231008
4.Others：This is a temporary script file.
"""

import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import math


def calc_class_weight(labels_count):
    """用于计算分类任务中的类别权重（class weights）。类别权重loss可以确保模型更多地关注少数类别。"""
    class_weight = dict()
    num_classes = len(labels_count)

    # 已知N1期最难分对。
    mu = [1, 1.5, 1, 1, 1]  # 5分类问题，可以再手动调整每类权重。

    tmp_max = 0
    for key in range(num_classes):
        tmp_max = max(tmp_max, labels_count[key])
        
    for key in range(num_classes): 
        class_weight[key] = math.sqrt(tmp_max / labels_count[key])
        class_weight[key] = round(class_weight[key] * mu[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]

    return class_weight


class MetricTracker:
    """主要功能是跟踪和记录指标，并计算它们的平均值。"""
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
