# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch import nn
from functools import reduce


def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model


# 根据mark[m]的位置得到其向量，作为mention的表示
def extract_mention_rep(mark_position, bert_rep):  # (B,1),(B,128,768)
    mention_rep = torch.tensor([]).to(mark_position.device)
    B = mark_position.size(0)
    # TODO：目前还是一条一条扫，看是否还能优化
    for i in range(B):
        embedding = torch.index_select(bert_rep[i], dim=0, index=mark_position.squeeze(1)[i])
        mention_rep = torch.cat([mention_rep, embedding])
    return mention_rep  # (B, 768)


class BertEncoder(nn.Module):
    def __init__(
        self, bert_model, output_dim, layer_pulled=-1, add_linear=None):
        super(BertEncoder, self).__init__()
        self.layer_pulled = layer_pulled
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        self.bert_model = bert_model
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def forward(self, token_ids, segment_ids, attention_mask, m_position=None):
        output_bert, output_pooler = self.bert_model(
            token_ids, segment_ids, attention_mask
        )
        # get embedding of [CLS] token
        if self.additional_linear is not None:
            embeddings = output_pooler

        # 以下两个else: (B, 128, 768) -> (B, 768)
        elif m_position is not None:  # 若给ctx编码，则取[m]位置对应的向量
            embeddings = extract_mention_rep(m_position, output_bert)  # (B, 768)
        else:
            embeddings = output_bert[:, 0, :]  # (B, 768)

        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings

        return result  # bi:[B*64, 768], cross:[B*64, 1]
