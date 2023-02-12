# -*- coding: utf-8 -*-
# Date       ：2022/6/22
# Author     ：Chen Xuekai
# Description：

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


def extract_mention_rep(mark_position, bert_rep):  # (B,1),(B,128,768)
    mention_rep = torch.tensor([]).to(mark_position.device)
    B = mark_position.size(0)
    # batch内逐条扫描得到mark词向量 TODO：目前还是一条一条扫，看是否还能优化
    for i in range(B):
        embedding = torch.index_select(bert_rep[i], dim=0, index=mark_position.squeeze(1)[i])
        mention_rep = torch.cat([mention_rep, embedding])
    return mention_rep  # (B, 768)


def to_bert_input(input_ids):
    attention_mask = 1 - (input_ids == 0).long()
    token_type_ids = torch.zeros_like(input_ids).long()
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }


def attention_layer(ctx_embedding, cand_embedding):
    ctx_embedding = ctx_embedding.unsqueeze(1)  # (B,1,768)
    atten_weights = torch.bmm(cand_embedding, ctx_embedding.transpose(2,1)).squeeze(2)  # (B,s_number,768)*(B,768,1)->(B,s_number)
    soft_atten_weights = F.softmax(atten_weights, 1)  # (B,s_number)
    new_cand_embedding = torch.bmm(cand_embedding.transpose(1, 2), soft_atten_weights.unsqueeze(2)).squeeze(2)  # (B,768,s_number)·(B,s_number,1)->(B,768)
    return new_cand_embedding  # (B,768)


class BiEncoder(nn.Module):
    def __init__(self, pretrained_model):
        super(BiEncoder, self).__init__()
        self.ctx_encoder = BertModel.from_pretrained(pretrained_model)
        self.ent_encoder = BertModel.from_pretrained(pretrained_model)

    def encode_context(self, ctx_ids, m_position):
        output_bert, output_pooler = self.ctx_encoder(**to_bert_input(ctx_ids))
        embeddings = extract_mention_rep(m_position, output_bert)  # (B, 768)
        return embeddings

    def encode_candidate(self, ent_ids):  # (B,s_number=10,seq_len=40)
        if len(ent_ids.size()) == 1:  # only one sentence
            ent_ids = ent_ids.unsqueeze(0)
            batch_size, sentence_num = 1, 1
        elif len(ent_ids.size()) == 3:  # TODO: ???
            batch_size, sentence_num, ent_seq_len = ent_ids.size()
            ent_ids = ent_ids.view(-1, ent_seq_len)
        else:
            batch_size, ent_seq_len = ent_ids.size()
            sentence_num = 1

        start_ids = 0
        candidate_output = []

        output_bert, output_pooler = self.ent_encoder(**to_bert_input(ent_ids))
        embeddings = output_bert[:, 0, :]  # 以下两个else: (B, 128, 768) -> (B, 768)
        # embeddings = output_bert  # (B, 768)，最后一层[CLS]经过Linear层和激活函数Tanh()后的向量
        return embeddings  # 期望输出(B,s_num,768)即每个view的[CLS]

    def score_candidate(self, text_vecs, cand_vecs, m_position, ent_emb=None):
        ctx_emb = self.encode_context(text_vecs, m_position)
        if ent_emb is None:  # 训练
            cand_emb = self.encode_candidate(cand_vecs)  # (B,s_number,768)
            cand_emb = attention_layer(ctx_emb, cand_emb)  # (B,768)
            return ctx_emb.mm(cand_emb.t())  # (B,768)*(768,B)->(B=16,B=16)
        else:  # 预测 cand_encs:(10000,768)
            return ctx_emb.mm(ent_emb.t())  # (1,768)*(768,10000)->(1,10000)

    def forward(self, ctx_ids, ent_ids, m_position):
        ctx_output = self.encode_context(ctx_ids, m_position)  # batch_size * hidden_size
        ent_output = self.encode_candidates(ent_ids)  # (batch_size * sentence_num) * hidden_size
        return ctx_output.contiguous(), ent_output.contiguous()
