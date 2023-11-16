"""完成N折的训练验证后，统计模型的平均结果"""
import os
import numpy as np

all_pres = []
all_labels = []
pres_list = []
labels_list = []
save_dir = "/home/xingjian.zhang/sleep/4_save/04_ST_FPZ-Cz&EOG/AttnSleep_2CH_S1/16_11_2023_12_12_fold9"
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

print("OK")
