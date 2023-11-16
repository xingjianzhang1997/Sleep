# -*- coding: utf-8 -*-
"""
1.Function：训练睡眠分期的主函数
2.Author：xingjian.zhang
3.Time：202310
4.Others：
1）实现睡眠分期的主函数  
2）引入weights-loss flag.
3) 增加并行双卡模型训练模式
"""

import json
from data_loader.data_loaders import *
from trainer import Trainer
from utils.util import calc_class_weight
import utils.metric as module_metric
import utils.loss as module_loss
import torch
import torch.nn as nn
import torch.optim as optim

import model.model as module_arch
import model.model_2CH as module_arch_2CH


# 为了可重复性，固定pytorch和numpy的随机种子
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.deterministic = True  # 让CuDNN也表现出确定性
else:
    device = torch.device('cpu')


def weights_init_normal(m):
    """模型的参数初始化"""
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        # torch.nn.init.kaiming_uniform_
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def print_counts(data_count):
    """打印样本量"""
    print("W期数据：{}".format(data_count[0]))
    print("N1期数据：{}".format(data_count[1]))
    print("N2期数据：{}".format(data_count[2]))
    print("N3期数据：{}".format(data_count[3]))
    print("REM期数据：{}".format(data_count[4]))
    print("所有数据总量：{}".format(sum(data_count)))


def select_model(name):
    """选择使用的模型"""
    model = module_arch.AttnSleep()  # 默认模型

    if name == "AttnSleep":
        model = module_arch.AttnSleep()
    elif name == "AttnSleep_2CH_S1":
        model = module_arch_2CH.AttnSleep_2CH_S1()
    elif name == "AttnSleep_2CH_S2":
        model = module_arch_2CH.AttnSleep_2CH_S2()
    else:
        print("当前选择的模型结构还未完成，强制使用默认模型")

    return model


def main(fold_id):
    batch_size = config["data_loader"]["args"]["batch_size"]
    learning_rate = config["optimizer"]["args"]["lr"]
    weight_decay = config["optimizer"]["args"]["weight_decay"]
    ams_grad = config["optimizer"]["args"]["ams_grad"]  # 修正Adam优化器

    # 构建AI模型和参数初始化
    model_name = config['arch']['type']
    model = select_model(model_name)
    model.apply(weights_init_normal)

    # getattr用于获取对象的属性或方法
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # 过滤出那些 requires_grad 属性为 True 的参数，即可训练的参数,通过优化器进行训练。
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay, amsgrad=ams_grad)

    data_loader, valid_data_loader, data_count = data_generator_np(folds_data[fold_id][0],
                                                                   folds_data[fold_id][1], batch_size)

    print("当前fold的编号：{} （从0开始）".format(fold_id))
    print_counts(data_count)

    print("*****开始训练*****")
    if flag_weights_loss:
        print("使用类别感知loss")
        weights_for_each_class = calc_class_weight(data_count)
    else:
        print("不使用类别感知loss")
        # 为了不改变后续代码结构，直接让每个类别的权重都一样
        weights_for_each_class = [0.2, 0.2, 0.2, 0.2, 0.2]

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      fold_id=fold_id,
                      data_count=data_count,
                      valid_data_loader=valid_data_loader,
                      class_weights=weights_for_each_class)

    early_stop_fold = config["early_stop_fold"]
    trainer.train(early_stop_fold=early_stop_fold)


if __name__ == '__main__':

    with open("config.json") as f:
        config = json.load(f)

    npz_file = config["npz_file"] + config["name"]
    num_folds = config["data_loader"]["args"]["num_folds"]

    flag_weights_loss = False  # loss训练是否使用类别感知。
    
    # 将npz数据全部读取，并且划分为num_folds组数据.
    if "sleep" in config["npz_file"]:
        folds_data = load_folds_data_sleep(npz_file, num_folds)
    elif "apples" in config["npz_file"]:
        folds_data = load_folds_data_apples(npz_file, num_folds)
    else:
        print("数据加载方式不匹配")

    # 10折交叉验证
    for cur_fold_id in range(num_folds):
        main(cur_fold_id)
