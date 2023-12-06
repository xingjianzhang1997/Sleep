"""
1.Function：基于pytorch的DataLoader进行数据加载
2.Author：xingjian.zhang
3.Time：20231007
4.Others：N.A.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import IterableDataset
import numpy as np
import os
from glob import glob


class MapDataset(Dataset):
    """Map式数据集: 将整个数据集读取到内存中，通过index映射的方式读取对应的数据，优点速度快，缺点占用内存，大的数据集是无法使用。"""
    def __init__(self, np_dataset):
        super(MapDataset, self).__init__()
        X_train = np.load(np_dataset[0])["x"]
        y_train = np.load(np_dataset[0])["y"]

        num = 0
        for np_file in np_dataset[1:]:
            num = num + 1
            print("完成加载的npz文件数量:{}".format(num))

            X_train = np.vstack((X_train, np.load(np_file)["x"]))  # np.vstack垂直堆叠这两个数组
            y_train = np.append(y_train, np.load(np_file)["y"])

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()

        # 保证input是 (sampple_size, #channels, seq_len)
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:  # 通常x_data.shape[1]是epoch*frequency，并不是1
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator_np(train_files, test_files, batch_size):

    train_dataset = MapDataset(train_files)  # train_files是文件地址
    test_dataset = MapDataset(test_files)

    # 计算每一类别的样本总数量（含训练+测试），这里有可改进的地方，因为counts同时被用来计算类别感知loss的权重了。
    all_ys = np.concatenate((train_dataset.y_data, test_dataset.y_data))
    all_ys = all_ys.tolist()  # 用于将pytorch的张量值转换为常规的Python列表
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                             num_workers=0)

    return train_loader, test_loader, counts


def load_folds_data_sleep(np_data_path, n_folds):
    """sleep数据集的加载"""
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    print("npz文件的数量：{}".format(len(files)))

    # files_dict的每个键是一个文件编号，对应的值是一个包含具有相同编号的文件路径的列表
    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1]
        file_num = file_name[3:6]  # 例如"ST7011J0.npz"，取编号为011

        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            # 将具有相同编号的文件路径分组在一起
            print(file_num)
            files_dict[file_num].append(i)

    files_pairs = []
    for key in files_dict:
        files_pairs.append(files_dict[key])
    files_pairs = np.array(files_pairs)

    # 将文件路径对按照交叉验证（cross-validation）的方式划分为不同的训练集和验证集
    train_files = np.array_split(files_pairs, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        subject_files = [item for sublist in subject_files for item in sublist]  # 它的作用就是将嵌套的列表展开成扁平的列表。
        files_pairs2 = [item for sublist in files_pairs for item in sublist]
        training_files = list(set(files_pairs2) - set(subject_files))  # subject_files是验证的那个fold
        folds_data[fold_id] = [training_files, subject_files]  # 以10折为例，每1次实验都是前9个fold是训练，最后1个fold是验证。
    return folds_data


def load_folds_data_apples(np_data_path, n_folds):
    """apples数据集的加载"""
    print("读取文件地址: {}".format(np_data_path))
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    temp_idx = 500
    files = files[0:temp_idx]
    print("npz文件的数量：{}".format(len(files)))
    # 将文件路径对按照交叉验证（cross-validation）的方式划分为不同的训练集和验证集
    train_files = np.array_split(files, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        training_files = list(set(files) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]  # 以10折为例，每1次实验都是前9个fold是训练，最后1个fold是验证。
    return folds_data

