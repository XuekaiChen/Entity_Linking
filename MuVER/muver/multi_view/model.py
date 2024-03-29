#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2021/03/16 15:32:52
@Author  :   Xinyin Ma
@Version :   0.1
@Contact :   maxinyin@zju.edu.cn
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from muver.utils.multigpu import GatherLayer
from transformers import BertModel, BertConfig


class BiEncoder(nn.Module):
    def __init__(self, pretrained_model):
        super(BiEncoder, self).__init__()
        self.ctx_encoder = BertModel.from_pretrained(pretrained_model, output_attentions=True, return_dict=True)
        self.ent_encoder = BertModel.from_pretrained(pretrained_model, output_attentions=True, return_dict=True)

    def to_bert_input(self, input_ids):
        attention_mask = 1 - (input_ids == 0).long()
        token_type_ids = torch.zeros_like(input_ids).long()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }

    def encode_context(self, ctx_ids, ctx_mask=None):
        if len(ctx_ids.size()) == 1:  # only one sentence
            ctx_ids = ctx_ids.unsqueeze(0)
        model_output = self.ctx_encoder(**self.to_bert_input(ctx_ids))
        context_output = model_output.last_hidden_state
        if ctx_mask is None:
            context_output = context_output[:, 0, :]
        else:
            context_output = torch.bmm(ctx_mask.unsqueeze(1), context_output).squeeze(1)
        return context_output

    def attention_layer(self, ctx_embedding, cand_embedding):  # (B,768),(B,s_number,768)
        ctx_embedding = ctx_embedding.unsqueeze(1)  # (B,1,768)
        atten_weights = torch.bmm(cand_embedding, ctx_embedding.transpose(2, 1)).squeeze(2)  # (B,s_number,768)*(B,768,1)->(B,s_number)
        soft_atten_weights = F.softmax(atten_weights, 1)  # (B,s_number)
        new_cand_embedding = torch.bmm(cand_embedding.transpose(1, 2), soft_atten_weights.unsqueeze(2)).squeeze(2)  # (B,768,s_number)·(B,s_number,1)->(B,768)
        return new_cand_embedding  # (B,768)

    def encode_candidates(self, ctx_output, ent_ids, ent_mask=None, interval=500, mode='train', view_expansion=True,
                          top_k=0.4, merge_layers=3):
        if len(ent_ids.size()) == 1:
            ent_ids = ent_ids.unsqueeze(0)
            batch_size, sentence_num = 1, 1
        elif len(ent_ids.size()) == 3:  # test中 如(1,6,40)
            batch_size, sentence_num, ent_seq_len = ent_ids.size()
            ent_ids = ent_ids.view(-1, ent_seq_len)
        else:
            batch_size, ent_seq_len = ent_ids.size()
            sentence_num = 1

        start_ids = 0
        candidate_output = []
        while start_ids < sentence_num * batch_size:  #
            model_output = self.ent_encoder(**self.to_bert_input(ent_ids[start_ids:start_ids + interval]))
            # last_hidden_state:(s_number,40,768), pooler_outpu每个CLS经过全链接的结果:t(s_number,768)
            candidate_output.append(model_output.last_hidden_state[:, 0, :])  # [batch_size, seq_len, hidden_size]
            start_ids += interval
        candidate_output = torch.cat(candidate_output, 0).view(batch_size, sentence_num, -1)  # (batch_size,10,768)
        candidate_output = self.attention_layer(ctx_output, candidate_output)  # (B,768)
        # if view_expansion:
        #     ori_views = candidate_output
        #     ori_ent_ids = ent_ids.view(batch_size, sentence_num, -1).tolist()
        #     new_pools = []
        #
        #     def merge_sequence(seq, ent_ids):
        #         s = ent_ids[seq[0]]
        #         sentence = s[:s.index(102)]
        #
        #         for i in range(1, len(seq)):
        #             s = ent_ids[seq[i]]
        #             mid_sentence = s[s.index(3) + 1:s.index(102)]
        #             if 0 in mid_sentence:
        #                 mid_sentence = mid_sentence[:mid_sentence.index(0)]
        #             sentence += mid_sentence
        #         if len(sentence) > 511:
        #             sentence = sentence[:511]
        #         sentence += [102]
        #         return sentence
        #
        #     def batch_sentences(sentences):
        #         max_len = max([len(s) for s in sentences])
        #         s_tensor = torch.zeros((len(sentences), max_len), dtype=torch.int64)
        #         for i, s in enumerate(sentences):
        #             s_tensor[i, :len(s)] = torch.tensor(s)
        #         return s_tensor
        #
        #     top_k = [top_k for _ in range(merge_layers)]
        #     target_sentence_num = int(sum(top_k) * sentence_num)
        #     for ori_view, ori_ent_id in zip(ori_views, ori_ent_ids):  # 遍历batch中样本
        #         new_pool = []
        #         views, seq_ids = ori_view, [[i] for i in range(len(ori_ent_id))]
        #
        #         for layer in range(len(top_k)):
        #             new_views, new_seq_ids = [], []
        #             dis = torch.sum(F.mse_loss(views[:-1], views[1:], reduction='none'), -1)
        #             # 需要融合的句子编号，个数为top_k*sent_num，如[0,1,2,5]
        #             merge_ids = torch.sort(dis, descending=True).indices[: int(views.size(0) * top_k[layer])].tolist()
        #
        #             new_sentences = []
        #             for i in range(len(seq_ids)):
        #                 if i in merge_ids and ori_ent_id[i][0] != 0:
        #                     seq_ids_a, seq_ids_b = seq_ids[i], seq_ids[i + 1]
        #                     seq_ids_merge = [] + seq_ids_a
        #                     for ids, ids_b in enumerate(seq_ids_b):
        #                         if ids_b > seq_ids_a[-1] and ori_ent_id[ids_b][0] != 0:
        #                             seq_ids_merge += [ids_b]
        #
        #                     if len(seq_ids_merge) != len(seq_ids_a):
        #                         new_sentence = merge_sequence(seq_ids_merge, ori_ent_id)
        #                         new_seq_ids.append(seq_ids_merge)
        #
        #                         new_sentences.append(new_sentence)
        #                         # new_repr = self.ent_encoder(**self.to_bert_input(new_sentence.unsqueeze(0).cuda())).last_hidden_state[:, 0, :]
        #                         # new_pool.append(new_repr)
        #                         new_views.append(None)
        #                     else:
        #                         new_seq_ids.append(seq_ids[i])
        #                         new_views.append(views[i])
        #                 else:
        #                     new_seq_ids.append(seq_ids[i])
        #                     new_views.append(views[i])
        #
        #             if len(new_sentences) > 0:
        #                 new_sentences = batch_sentences(new_sentences)
        #                 pool = []
        #                 start_ids = 0
        #                 while start_ids < new_sentences.size(0):
        #                     new_repr = self.ent_encoder(
        #                         **self.to_bert_input(new_sentences[start_ids:start_ids + 50].cuda())).last_hidden_state[
        #                                :, 0, :]
        #                     pool += new_repr
        #                     start_ids += 50
        #                 new_repr = pool
        #                 new_pool += new_repr
        #                 view_idx = 0
        #                 for i, new_view in enumerate(new_views):
        #                     if new_view is None:
        #                         new_views[i] = new_repr[view_idx]
        #                         view_idx += 1
        #                 assert view_idx == len(new_repr)
        #             views, seq_ids = torch.stack(new_views, 0), new_seq_ids
        #
        #         if mode == 'train':
        #             if len(new_pool) == 0:
        #                 new_pool = ori_view[-1].unsqueeze(0).repeat(target_sentence_num, 1)
        #             else:
        #                 new_pool = torch.stack(new_pool, 0)
        #                 if len(new_pool) < target_sentence_num:
        #                     new_pool = torch.cat(
        #                         [new_pool, ori_view[-1].unsqueeze(0).repeat(target_sentence_num - len(new_pool), 1)], 0)
        #
        #         if mode == 'train':
        #             new_pools.append(new_pool)
        #         else:
        #             new_pools += new_pool
        #     if mode == 'train':
        #         new_pools = torch.stack(new_pools, 0)
        #         # print(new_pools.shape, ori_views.shape)
        #         candidate_output = torch.cat([ori_views, new_pools], 1)
        #     else:
        #         if len(new_pools) > 0:
        #             new_pools = torch.stack(new_pools, 0)
        #             candidate_output = torch.cat([candidate_output.squeeze(0), new_pools], 0).unsqueeze(
        #                 0)  # (B,merge_layers*layers,768)
        return candidate_output  # view:(B,12,768); w/o_view:(B,10,768); att:(B,768)

    def score_candidates(self, ctx_ids, ctx_world, ctx_mask=None, candidate_pool=None):
        # candidate_pool: (entity_num * 9) * hidden_state
        ctx_output = self.encode_context(ctx_ids, ctx_mask).cpu().detach()
        res = []
        for world, ctx_repr in zip(ctx_world, ctx_output):
            ctx_repr = ctx_repr.to(candidate_pool[world].device)
            res.append(ctx_repr.unsqueeze(0).mm(candidate_pool[world].T).squeeze(0))
        return res

    def forward(self, ctx_ids, ent_ids):
        ctx_output = self.encode_context(ctx_ids)  # batch_size * 768
        ent_output = self.encode_candidates(ctx_output, ent_ids)  # batch_size * 768
        scores = ctx_output.mm(ent_output.t())  # batch_size * batch_size
        predict = torch.max(scores, -1).indices

        bs = scores.size(0)
        target = torch.LongTensor(torch.arange(bs))
        target = target.to(ctx_output.device)

        loss = F.cross_entropy(scores, target, reduction="mean")
        acc = sum(predict == target) * 1.0 / scores.size(0)
        return loss, acc, scores

