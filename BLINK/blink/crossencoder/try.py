# -*- coding: utf-8 -*-
# Date        : 2022/3/7
# Author      : Chen Xuekai
# Description :

import torch
import os

data_path = "/data/chenxuekai/Entity_Linking/BLINK/models/zeshel/top64_candidates/"

fname1 = os.path.join(data_path, "train.t7")
train_data = torch.load(fname1)
context_input = train_data["context_vecs"]
candidate_input = train_data["candidate_vecs"]
label_input = train_data["labels"]
print("context_input", context_input.shape)
print("candidate_input", candidate_input.shape)
print("label_input", label_input.shape)
print(torch.max(label_input, 0))
print(torch.min(label_input, 0))

for k in train_data.keys():
    print(k)

src_input = train_data['worlds']
print("src_input", src_input.shape)
print(src_input)

# print("valid")
# fname2 = os.path.join(data_path, "valid.t7")
# valid_data = torch.load(fname2)
# context_input2 = valid_data["context_vecs"]
# candidate_input2 = valid_data["candidate_vecs"]
# label_input2 = valid_data["labels"]
# print("context_input", context_input2.shape)
# print("candidate_input", candidate_input2.shape)
# print("label_input", label_input2.shape)
# for k in valid_data.keys():
#     print(k)
#
#
# print("test")
# fname3 = os.path.join(data_path, "test.t7")
# test_data = torch.load(fname3)
# context_input3 = test_data["context_vecs"]
# candidate_input3 = test_data["candidate_vecs"]
# label_input3 = test_data["labels"]
# print("context_input", context_input3.shape)
# print("candidate_input", candidate_input3.shape)
# print("label_input", label_input3.shape)
# for k in test_data.keys():
#     print(k)

from ner_model import MyNERModel
type_model_path = "data\\chenxuekai\\Entity_Linking\\ner4el\\ner4el\\wandb\\ner_classifier.pt"
ner_model = MyNERModel.gpu()
ner_model.load_state_dict(torch.load(type_model_path))
