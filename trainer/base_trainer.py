"""
1.Function：基本的训练框架，会读取很多config.json文件的值
2.Author：xingjian.zhang
3.Time：20231007实现V1版本
4.Others：V2-精简代码  20231113
"""

import torch
from datetime import datetime
from abc import abstractmethod  # 可以定义抽象类（只声明，不实现）
from numpy import inf  # inf: 表示正无穷
import numpy as np
import pandas as pd
import xlwt
import os
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


class BaseTrainer:
    """训练的基础通用代码"""
    def __init__(self, model, criterion, metric_ftns, optimizer, config, fold_id, data_count):
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.config = config
        self.fold_id = fold_id
        self.data_count = data_count  # 含每个类别的数据量

        self.epochs = config['trainer']['epochs']
        self.save_period = config['trainer']['save_period']  # 保存训练检查点的epoch间隔

        self.start_epoch = 1

        # 数据保存地址
        self.save_dir = config['trainer']['save_dir']
        self.exper_name = config['name']
        self.model_name = config['arch']['type']
        run_id = datetime.now().strftime('%d_%m_%Y_%H_%M') + "_fold" + str(fold_id)
        self.checkpoint_dir = self.save_dir + "/" + self.exper_name + "/" + self.model_name + "/" + run_id + "/"
        os.makedirs(self.checkpoint_dir, exist_ok=True)  # 创建目录，如果不存在

        # 设置使用的GPU
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        print("使用的显卡数量： {}".format(len(device_ids)))
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)  # 显卡数量大于1，则分布式训练

        # 设置保存最佳模型的监测方式
        self.monitor = config['trainer']['monitor']
        assert self.monitor in ["loss", "acc"]  # 检查monitor的输入是否符合格式
        self.mnt_best = inf if self.monitor == "loss" else -inf

        # resume恢复之前保存的模型或训练状态
        if config["resume_path"]:
            print("resume的地址：{}".format(config["resume_path"]))
            self._resume_checkpoint(config["resume_path"])

    def train(self, early_stop_fold=False):
        """训练的核心代码"""
        best_pres = []
        labels = []
        results_xls = self._create_excel()

        for epoch in range(self.start_epoch, self.epochs + 1):
            # 模型训练与验证。
            train_result, best_pres, labels = self._train_epoch(epoch)

            # 打印和保存训练/验证的loss/ACC，保存模型直接结果。
            results_xls = self._write_excel(results_xls, epoch, train_result)

            # 选择保存最佳的模型
            best_model_path = self.checkpoint_dir + 'checkpoint_best.pth'
            self._check_best_model(train_result, epoch, best_model_path)

            # 周期保存模型
            if epoch % self.save_period == 0:
                periodic_model_path = self.checkpoint_dir + 'checkpoint_epoch{}.pth'.format(epoch)
                self._save_checkpoint(periodic_model_path, epoch)

        # 每个fold新建一个excel表，存储每个epoch的训练集/验证集的loss和ACC结果，存储模型直接结果
        self._save_result(results_xls, best_pres, labels)

        if self.fold_id == self.config["data_loader"]["args"]["num_folds"] - 1:
            self._calc_metrics()  # 计算平均结果

        # 提前结束，只训练2折
        if early_stop_fold:
            num_fold = 2
            if self.fold_id == num_fold - 1:
                print("提前结束，只训练验证了2折")
                self._calc_metrics()
                sys.exit()

    @staticmethod
    def _prepare_device(n_gpu_use):
        """配置GPU的使用"""
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("警告: 没有可用的GPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print("警告: 设置{}个GPU使用，但是只有{}个GPU是可用的".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _resume_checkpoint(self, resume_path):
        """加载和恢复Checkpoint path"""
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # 加载模型的结构.
        if checkpoint['config']['arch'] != self.config['arch']:
            print("警告！！！ checkpoint保存的模型结构和配置文件中加载的模型结构不同，可能会导致异常。")
        self.model.load_state_dict(checkpoint['state_dict'])

        # 加载优化器的参数.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            print("警告！！！checkpoint保存的优化器类型和配置文件中的优化器类型不同，可能会导致异常。")
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint文件加载完毕. 从epoch {}开始继续训练".format(self.start_epoch))

    def _calc_metrics(self):
        """完成N折的训练验证后，统计模型的平均结果"""
        all_pres = []
        all_labels = []
        pres_list = []
        labels_list = []
        save_dir = os.path.abspath(os.path.join(self.checkpoint_dir, os.pardir))  # 获取上级目录的绝对路径
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                if "pres" in file:
                    pres_list.append(os.path.join(root, file))
                if "labels" in file:
                    labels_list.append(os.path.join(root, file))

        print("最终结果由 {}折 组成".format(len(pres_list)))

        for i in range(len(pres_list)):
            all_pres.extend(np.load(pres_list[i]))
            all_labels.extend(np.load(labels_list[i]))

        all_labels = np.array(all_labels).astype(int)
        all_pres = np.array(all_pres).astype(int)

        r = classification_report(all_labels, all_pres, digits=5, output_dict=True)
        cm = confusion_matrix(all_labels, all_pres)
        df = pd.DataFrame(r)
        df["cohen"] = cohen_kappa_score(all_labels, all_pres)
        df["accuracy"] = accuracy_score(all_labels, all_pres)
        df = df * 100
        file_name = self.config["name"] + "_classification_report.xlsx"
        report_Save_path = os.path.join(save_dir, file_name)
        df.to_excel(report_Save_path)
        cm_file_name = self.config["name"] + "_confusion_matrix.torch"
        cm_Save_path = os.path.join(save_dir, cm_file_name)
        torch.save(cm, cm_Save_path)

    def _check_best_model(self, result, epoch, best_model_path):
        """选择最佳的模型"""
        epoch_ValLoss = result['val_loss']
        epoch_ValAcc = result['val_accuracy']

        # 要求最佳的模型一定要至少训练10个epoch
        if epoch > 10:
            if self.monitor == "loss":
                if epoch_ValLoss < self.mnt_best:
                    self.mnt_best = epoch_ValLoss
                    self._save_checkpoint(best_model_path, epoch)
            elif self.monitor == "acc":
                if epoch_ValAcc > self.mnt_best:
                    self.mnt_best = epoch_ValAcc
                    self._save_checkpoint(best_model_path, epoch)
            else:
                pass

    def _save_checkpoint(self, path_name, epoch):
        """保存模型"""
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        torch.save(state, path_name)
        print("Saving checkpoint: {} ...".format(path_name))

    def _save_result(self, excel, pres, labels):
        """数据保存到本机地址"""
        excelResultPath = self.checkpoint_dir + 'Result_fold{}.xlsx'.format(self.fold_id)
        presPath = self.checkpoint_dir + 'All_pres_fold' + str(self.fold_id)
        labelsPath = self.checkpoint_dir + 'All_labels_fold' + str(self.fold_id)
        excel.save(excelResultPath)
        np.save(presPath, pres)
        np.save(labelsPath, labels)

    def _create_excel(self):
        """创建一个含表头的excel"""
        xls = xlwt.Workbook()

        sht1 = xls.add_sheet("Result")
        sht1.write(0, 0, 'epoch')
        sht1.write(0, 1, 'Train_loss')
        sht1.write(0, 2, 'Train_ACC')
        sht1.write(0, 3, 'Val_loss')
        sht1.write(0, 4, 'Val_ACC')

        sht2 = xls.add_sheet("Data")
        sht2.write(0, 0, 'W')
        sht2.write(0, 1, 'N1')
        sht2.write(0, 2, 'N2')
        sht2.write(0, 3, 'N3')
        sht2.write(0, 4, 'REM')
        sht2.write(0, 5, 'ALL')
        sht2.write(1, 0, self.data_count[0])
        sht2.write(1, 1, self.data_count[1])
        sht2.write(1, 2, self.data_count[2])
        sht2.write(1, 3, self.data_count[3])
        sht2.write(1, 4, self.data_count[4])
        sht2.write(1, 5, sum(self.data_count))

        return xls

    @staticmethod
    def _write_excel(xls, cur_epoch, result):
        """保存和打印训练过程中的结果值到已生成的excel中"""
        sht1 = xls.get_sheet("Result")

        tmp_result_list = []
        for key, value in result.items():
            tmp_result_list.append(value)
        epoch_TrainLoss, epoch_TrainACC, epoch_ValLoss, epoch_ValACC = map(lambda x: '%.3f' % x, tmp_result_list[0:4])

        index_XLS = cur_epoch
        sht1.write(index_XLS, 0, cur_epoch)
        sht1.write(index_XLS, 1, epoch_TrainLoss)
        sht1.write(index_XLS, 2, epoch_TrainACC)
        sht1.write(index_XLS, 3, epoch_ValLoss)
        sht1.write(index_XLS, 4, epoch_ValACC)

        print("当前Epoch：{}".format(cur_epoch))
        print("当前Train_loss：{}".format(epoch_TrainLoss))
        print("当前Train_Acc：{}".format(epoch_TrainACC))
        print("当前Val_loss：{}".format(epoch_ValLoss))
        print("当前Val_Acc：{}".format(epoch_ValACC))

        return xls

    @abstractmethod
    def _train_epoch(self, epoch):
        """先定义一个函数，用来写每个epoch的训练逻辑，epoch是当前的epoch序号"""
        raise NotImplementedError  # 当子类没有提供具体实现时，调用这个抽象方法将会引发异常
