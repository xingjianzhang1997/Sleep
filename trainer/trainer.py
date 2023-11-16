# -*- coding: utf-8 -*-
"""
1.Function：模型的训练框架
2.Author：xingjian.zhang
3.Time：20231008
4.Others：This is a temporary script file.
"""

import numpy as np
import torch
from base_trainer import BaseTrainer
from utils import MetricTracker
from torch.optim.lr_scheduler import StepLR

selected_d = {"pres": [], "labels": []}


class Trainer(BaseTrainer):
    """Trainer class """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, fold_id, data_count, 
                 valid_data_loader=None, class_weights=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, fold_id, data_count)
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

        self.fold_id = fold_id
        self.selected_acc = 0
        self.selected_loss = 100
        self.class_weights = class_weights
        self.data_count = data_count  # 每个类别的数据量

        self.step_size = config["StepLR"]["step_size"]
        self.gamma = config["StepLR"]["gamma"]
        # 创建StepLR学习率调度器，每step_size个epoch，学习率衰减*gamma
        self.scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

    def _train_epoch(self, epoch):
        """每个epoch的训练函数，同时会调用验证函数"""
        self.model.train()
        self.train_metrics.reset()
        
        for batch_idx, (data, label) in enumerate(self.data_loader):
            data, label = data.to(self.device), label.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.criterion(output, label, self.class_weights, self.device)
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, label))

            if batch_idx == self.len_epoch:
                break

        # 打印当前epoch的学习率
        # current_lr = self.optimizer.param_groups[0]['lr']
        # print(f"Epoch {epoch}, Current Learning Rate: {current_lr}")

        # 在每个epoch结束后更新学习率调度器
        self.scheduler.step()
        log = self.train_metrics.result()

        if self.do_validation:
            val_log, pres, labels = self._valid_epoch()
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.config["trainer"]["monitor"] == "loss":
                if val_log["loss"] < self.selected_loss:
                    self.selected_loss = val_log["loss"]
                    selected_d["pres"] = pres
                    selected_d["labels"] = labels
            elif self.config["trainer"]["monitor"] == "acc":
                if val_log["accuracy"] > self.selected_acc:
                    self.selected_acc = val_log["accuracy"]
                    selected_d["pres"] = pres
                    selected_d["labels"] = labels

        return log, selected_d["pres"], selected_d["labels"]

    def _valid_epoch(self):
        """每训练完1个epoch后，验证集完整验证1次模型"""
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            pres = np.array([])
            labels = np.array([])
            for batch_idx, (data, label) in enumerate(self.valid_data_loader):
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, label, self.class_weights, self.device)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, label))

                prediction = output.data.max(1, keepdim=True)[1].cpu()

                pres = np.append(pres, prediction.cpu().numpy())
                labels = np.append(labels, label.data.cpu().numpy())

        return self.valid_metrics.result(), pres, labels
    