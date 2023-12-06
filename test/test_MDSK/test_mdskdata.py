# -*- coding: utf-8 -*-
"""
1.Function：加载训练好的checkpoint参数和模型结构，跨数据集进行推理
2.Author：xingjian.zhang
3.Time：2023年11月
4.Others：1）双通道的模型 脑电+眼电推理
          2）MDSK数据没有标签，直接推理。
"""

import torch
import os
import pandas as pd
import numpy as np
import model.model_2CH_1 as module_arch
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from torch.utils.data import Dataset
from glob import glob
from my_functions import *


class LoadDataset_from_numpy(Dataset):
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy, self).__init__()
        # load files
        X_train = np.load(np_dataset[0])["x"]
        y_train = np.load(np_dataset[0])["y"]
        for np_file in np_dataset[1:]:
            X_train = np.vstack((X_train, np.load(np_file)["x"]))  # np.vstack垂直堆叠这两个数组
            y_train = np.append(y_train, np.load(np_file)["y"])

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()  # 用于将张量的数据类型转换为64位整数（long）

        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)
        print("data的shape {}".format(self.x_data.shape))

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def loadTestData(path):
    """读取所有的测试"""
    files = sorted(glob(os.path.join(path, "*.npz")))
    print("npz文件的数量：{}".format(len(files)))
    # 数据变成pytorch的读取形式
    batch_size = 128
    test_dataset = LoadDataset_from_numpy(files)
    all_ys = test_dataset.y_data.tolist()  # 用于将pytorch的张量值转换为常规的Python列表
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]  # 读取每类样本的数据

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)
    return test_loader, counts


def loadModel(path):
    """加载模型结构和参数"""
    myModel = module_arch.AttnSleep_2CH_S2()
    state = torch.load(path)
    myModel.load_state_dict(state["state_dict"])
    return myModel


def testModel(myModel, Data):
    device = torch.device('cuda:0')
    myModel.eval()  # 进入推理模式
    with torch.no_grad():
        outs = np.array([])
        trgs = np.array([])
        for batch_idx, (data, target) in enumerate(Data):
            data = downSample(data)
            data = data.to(device)
            myModel = myModel.to(device)
            output = myModel(data)
            preds_ = output.data.max(1, keepdim=True)[1].cpu()
            outs = np.append(outs, preds_.numpy())
            trgs = np.append(trgs, target.data.numpy())

    return outs, trgs


def main():
    savePath = "/home/xingjian.zhang/sleep/3_independent_result/MDSK_ST_FPZ-Cz&EOG_model/"  # 测试结果保存地址
    checkpointPath = "/home/xingjian.zhang/sleep/4_save/sleep-ST/ST_FPZ-Cz&EOG/AttnSleep_2CH_S2_1/17_11_2023_09_49_fold0/checkpoint_best.pth"  # 调用模型参数
    npzDataPath = "/home/xingjian.zhang/sleep/0_data/99_mdskNPZdata"  # 测试数据地址
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    model = loadModel(checkpointPath)
    testData, numEachClasses = loadTestData(npzDataPath)
    y_pre, y_target = testModel(model, testData)  # 结果是numpy格式

    y_pre = np.array(y_pre).astype(int)
    save_file_path = os.path.join(savePath, "y_pre.npy")
    np.save(save_file_path, y_pre)

if __name__ == "__main__":
    main()
