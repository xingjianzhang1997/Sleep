
import torch
import os
import pandas as pd
import numpy as np
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
            print(np_file)
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


npzDataPath = "/home/xingjian.zhang/sleep/0_data/05_applesNPZdata"
testData, numEachClasses = loadTestData(npzDataPath)
print(numEachClasses)