"""
1.Function：loss计算函数
2.Author：xingjian.zhang
3.Time：20231007
4.Others：目前只写了一个损失函数，未来可以增加不同的损失函数进行对比。
"""

import torch
import torch.nn as nn


def weighted_CrossEntropyLoss(output, target, classes_weights, device):
    cr = nn.CrossEntropyLoss(weight=torch.tensor(classes_weights).to(device))
    return cr(output, target)
