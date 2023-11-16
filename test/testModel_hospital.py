#####
# function： 加载训练好的checkpoint和模型，基于医院的subject数据进行推理。
#####


import torch
import os
import pandas as pd
import numpy as np
import model.model as module_arch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
    # 获取模型结构
    myModel = module_arch.AttnSleep()
    # pint(myModel)
    # 加载模型参数
    state = torch.load(path)
    myModel.load_state_dict(state["state_dict"])

    return myModel


def downSample(data):
    flag = 1   # 0：均匀欠采， 1:抗混叠滤波器后对信号进行下采样，   其他：离散傅里叶下采样
    # print(data.shape)  # [batchSize, channel, points]
    if flag == 0:  # 均匀欠采
        all_index = np.arange(15360)  # 512 * 30

        # 将数据分成多个子组，每个子组包含512个点
        subgroup_size = 512
        subgroups = [all_index[i:i + subgroup_size] for i in range(0, len(all_index), subgroup_size)]

        # 对每个子组均匀采样为100个点
        sampled_data = []
        sample_size = 100
        for subgroup in subgroups:
            if len(subgroup) < sample_size:
                continue
            indices = np.linspace(0, len(subgroup) - 1, sample_size, dtype=int)
            sampled_subgroup = subgroup[indices]
            sampled_data.extend(sampled_subgroup)
        data = data[:, :, sampled_data]
    elif flag == 1:
        import random
        from scipy import signal
        data = data.numpy()
        q = 5  # 下采样的倍数
        data = signal.decimate(data, q, axis=-1)
        sample = sorted(random.sample(list(range(3072)), 3000))
        data = data[:, :, sample].copy()
        data = torch.tensor(data)
    else:
        from scipy import signal
        data = data.numpy()
        data = signal.resample(data, 3000, axis=-1)
        # data = signal.resample_poly(data, 100, 512, axis=-1)
        data = torch.tensor(data)  # copy操作可以在原先的numpy变量中创造一个新的不适用负索引的numpy变量

    return data


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


# 保存好的checkpoint
modelPath = "/home/xingjian.zhang/sleep/1_code/00_Saved/00_SC_FPZ-Cz/"
modelNames = os.listdir(modelPath)
checkpointPath = modelPath + modelNames[0] + "/checkpoint-epoch40.pth"

name = checkpointPath.split("/")[6]
# 数据保存地址
savePath = "/home/xingjian.zhang/sleep/3_independent_result/" + name
if not os.path.exists(savePath):
    os.makedirs(savePath)

npzDataPath = "/home/xingjian.zhang/sleep/0_data/03_hospitalNPZdata/"
files = sorted(glob(os.path.join(npzDataPath, "*.npz")))
print("subject npz文件的数量：{}".format(len(files)))
for i in range(len(files)):
    testName = files[i].split("/")[-1]
    testName = testName.split(".")[0]
    # 数据变成pytorch的读取形式
    batch_size = 128
    path = [files[i]]
    test_dataset = LoadDataset_from_numpy(path)
    all_ys = test_dataset.y_data.tolist()  # 用于将pytorch的张量值转换为常规的Python列表
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]  # 读取每类样本的数据
    testData = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    model = loadModel(checkpointPath)
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
    print(testName)
    print("准确率值为 {}%".format(round(acc*100, 2)))
    print("验证结束, 详细测试情况请见《{}》".format(file_name))
    print("*******************")
