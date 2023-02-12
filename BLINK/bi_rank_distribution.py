# -*- coding: utf-8 -*-
# Date        : 2022/4/7
# Author      : Chen Xuekai
# Description : 查看bi-encoder召回后gold的排名分布

import torch
import matplotlib.pyplot as plt

fname = "models/zeshel/all_sample_top64_candidates/test.t7"
test_data = torch.load(fname)
all_label_input = test_data['labels'].tolist()   # [10000]

# 统计频数
frequency_dict = {}
for sample in all_label_input:
    try:
        frequency_dict[sample] += 1
    except:
        frequency_dict[sample] = 1

no_recall = frequency_dict[-1]
frequency_dict[70] = no_recall
recall = len(all_label_input) - no_recall
del frequency_dict[-1]
# fig, ax = plt.subplots()
# save_name = "bi-encoder gold rank distribution"
# ax.set_title(save_name, fontsize=14)
# plt.bar(frequency_dict.keys(), frequency_dict.values())
# plt.savefig("{}.png".format(save_name))
bi_acc = 100 * frequency_dict[0] / recall
print("bi-accuracy: %.4f %%" % bi_acc)

