# -*- coding: utf-8 -*-
# Date       ：2022/6/17
# Author     ：Chen Xuekai
# Description：

import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from muver.utils.multigpu import GatherLayer
from transformers import BertModel, BertConfig


# Transformer Parameters
d_model = 512  # Embedding Size (include Word_embedding, Positional_embedding)
d_ff = 2048  # FeedForward dimension: Embedding_dim(512) --> d_ff(2048)
d_k = d_q = 64  # dimension of K(= Q), V
d_v = 64  # K可以不等于V
n_heads = 8  # number of heads in Multi-Head Attention


# Positional_Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        PE(POS, 2i) = sin(pos/10000^(2i/d_model))
        PE(POS, 2i+1) = cos(pos/10000^(2i/d_model))
        """
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        PE = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        PE[:, 0::2] = torch.sin(position * div_term)
        PE[:, 1::2] = torch.cos(position * div_term)
        PE = PE.unsqueeze(0).transpose(0, 1)
        self.register_buffer('PE', PE)

    def forward(self, Inputs):  # Inputs: [seq_len, batch_size, d_model]
        Inputs = Inputs + self.PE[: Inputs.size(0), :]
        return self.dropout(Inputs)


# Scaled Dot-Product Attention(Figure 2)
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]


        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model)(output + residual), attn


# FeedForward
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        residual = inputs  # inputs: [batch_size, seq_len, d_model]
        output = self.fc(inputs)
        return nn.LayerNorm(d_model)(output + residual)  # [batch_size, seq_len, d_model]


# EncoderLayer
class SentenceAttention(nn.Module):
    def __init__(self):
        super(SentenceAttention, self).__init__()
        self.pos_emb = PositionalEmbedding(d_model)
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_inputs = self.pos_emb(enc_inputs.transpose(0, 1)).transpose(0, 1)
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs


class BiEncoder(nn.Module):
    def __init__(self, pretrained_model):
        super(BiEncoder, self).__init__()
        self.ctx_encoder = BertModel.from_pretrained(pretrained_model, output_attentions=True, return_dict=True)
        self.ent_encoder = BertModel.from_pretrained(pretrained_model, output_attentions=True, return_dict=True)
        self.sentence_encoder = SentenceAttention()

    def to_bert_input(self, input_ids):
        attention_mask = 1 - (input_ids == 0).long()
        token_type_ids = torch.zeros_like(input_ids).long()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }

    def encode_candidates(self, ent_ids, ent_mask=None, interval=500, mode='train', view_expansion=True, top_k=0.4, merge_layers=3):
        if len(ent_ids.size()) == 1:  # only one sentence
            ent_ids = ent_ids.unsqueeze(0)
            batch_size, sentence_num = 1, 1
        elif len(ent_ids.size()) == 3:
            batch_size, sentence_num, ent_seq_len = ent_ids.size()
            ent_ids = ent_ids.view(-1, ent_seq_len)
        else:
            batch_size, ent_seq_len = ent_ids.size()
            sentence_num = 1

        start_ids = 0
        candidate_output = []
        while start_ids < sentence_num * batch_size:  # 仅仅为了防止进入bert的batch太大处理不了
            model_output = self.ent_encoder(**self.to_bert_input(ent_ids[start_ids:start_ids + interval]))
            candidate_output.append(model_output.last_hidden_state[:, 0, :])  # [batch_size, seq_len, hidden_size]
            start_ids += interval
        candidate_output = torch.cat(candidate_output, 0).view(batch_size, sentence_num, -1)  # (batch_size,10,768)
        candidate_output = self.sentence_encoder(candidate_output)
        return candidate_output

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

    def score_candidates(self, ctx_ids, ctx_world, ctx_mask=None, candidate_pool=None):
        # candidate_pool: (entity_num * 9) * hidden_state
        ctx_output = self.encode_context(ctx_ids, ctx_mask).cpu().detach()
        res = []
        for world, ctx_repr in zip(ctx_world, ctx_output):
            ctx_repr = ctx_repr.to(candidate_pool[world].device)
            res.append(ctx_repr.unsqueeze(0).mm(candidate_pool[world].T).squeeze(0))
        return res

    def forward(self, ctx_ids, ent_ids, num_gpus=0):
        '''
        if num_gpus > 1:
            ctx_ids = ctx_ids.to("cuda:0")
            ent_gpus = num_gpus - 1
            per_gpu_batch = ent_ids // ent_gpus

            ent_ids = ent_ids.to("cuda:1")
        elif num_gpus == 1:
            ctx_ids = ctx_ids.cuda()
            ent_ids = ent_ids.cuda()
        '''
        batch_size, sentence_num, ent_seq_len = ent_ids.size()

        ctx_output = self.encode_context(ctx_ids)  # batch_size * hidden_size
        ent_output = self.encode_candidates(ent_ids)  # (batch_size * sentence_num) * hidden_size
        return ctx_output.contiguous(), ent_output.contiguous()



