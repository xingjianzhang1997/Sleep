# -*- coding: utf-8 -*-
"""
1.Function：评估指标函数
2.Author：xingjian.zhang
3.Time：20231007
4.Others：目前只写了基本的评估函数，未来可以增加不同的评估函数，得出更全面的测试结果。
"""

import torch
from sklearn.metrics import confusion_matrix, f1_score


def accuracy(output, target):
    # 不更新梯度，用于模型推理阶段
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred==target).item()
    return correct / len(target)


# F1 分数是精确度（precision）和召回率（recall）的一种综合性能度量，用于衡量分类模型的性能。
# 'macro' 表示计算每个类别的 F1 分数，然后对它们取平均值。
# .cpu()方法将张量移到 CPU 上

def f1(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return f1_score(pred.cpu().numpy(), target.data.cpu().numpy(), average='macro')
