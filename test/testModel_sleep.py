# -*- coding: utf-8 -*-
"""
1.Function：加载训练好的checkpoint参数和模型结构，跨数据集进行推理
2.Author：xingjian.zhang
3.Time：2023年11月
4.Others：1）清洗不需要的数据，把剩余的数据和标签合并，处理为30S的epoch片段，存为npz文件.
          2）引入数据平衡算法，欠采和扩增。flag_data_balance控制
"""

import torch
import os
import pandas as pd
import numpy as np
import model.model as module_arch
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from torch.utils.data import Dataset
from glob import glob


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
        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def loadModel(path):
    # checkpoint保存的state字典内容
    # state = {
    #     'arch': arch,
    #     'epoch': epoch,
    #     'state_dict': self.model.state_dict(),
    #     'optimizer': self.optimizer.state_dict(),
    #     'monitor_best': self.mnt_best,
    #     'config': self.config
    # }

    # 获取模型结构
    myModel = module_arch.AttnSleep()
    # pint(myModel)

    # 加载模型参数
    state = torch.load(path)
    myModel.load_state_dict(state["state_dict"])

    return myModel


def loadTestData(path):
    # 读取所有数据
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


def testModel(myModel, Data):
    device = torch.device('cuda:0')
    myModel.eval()  # 进入推理模式
    with torch.no_grad():
        outs = np.array([])
        trgs = np.array([])
        for batch_idx, (data, target) in enumerate(Data):
            data = data.to(device)
            myModel = myModel.to(device)
            output = myModel(data)
            preds_ = output.data.max(1, keepdim=True)[1].cpu()
            outs = np.append(outs, preds_.numpy())
            trgs = np.append(trgs, target.data.numpy())

    return outs, trgs

# 结果保存地址
savePath = "/home/xingjian.zhang/sleep/3_independent_result/00_SC_FPZ-Cz_model/"    # 02_SC_EOG_model\00_SC_FPZ-Cz_model
if not os.path.exists(savePath):
    os.makedirs(savePath)

# 保存好的checkpoint
checkpointPath = "/home/xingjian.zhang/sleep/1_code/00_Saved/00_SC_FPZ-Cz/27_10_2023_10_07_26_fold6/checkpoint-epoch40.pth"

# 测试数据地址
allDataPath = "/home/xingjian.zhang/sleep/0_data/01_sleepNPZdata/"
testName = "00_ST_FPZ-Cz"  # 00_ST_FPZ-Cz\01_SC_FPZ-Cz\02_SC_EOG\03_ST_EOG
npzDataPath = os.path.join(allDataPath, testName)

model = loadModel(checkpointPath)
testData, numEachClasses = loadTestData(npzDataPath)
y_pre, y_target = testModel(model, testData)  # 结果是numpy格式

y_pre = np.array(y_pre).astype(int)
y_target = np.array(y_target).astype(int)

cr = classification_report(y_target, y_pre, output_dict=True)
cm = confusion_matrix(y_target, y_pre)
acc = accuracy_score(y_pre, y_target)
df = pd.DataFrame(cr)
df["accuracy"] = acc
df = df * 100
file_name = testName + "_classification_report.xlsx"
report_Save_path = os.path.join(savePath, file_name)
df.to_excel(report_Save_path)

cm_file_name = testName + "_confusion_matrix.torch"
cm_Save_path = os.path.join(savePath, cm_file_name)
torch.save(cm, cm_Save_path)
print("准确率值为 {}%".format(round(acc*100, 2)))
print("验证结束, 详细测试情况请见《{}》".format(file_name))
