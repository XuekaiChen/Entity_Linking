# -*- coding: utf-8 -*-
# Date        : 2022/4/9
# Author      : Chen Xuekai
# Description :
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)

from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.common.ranker_base import BertEncoder, get_model_obj
from blink.common.optimizer import get_bert_optimizer


def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModule, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        cand_bert = BertModel.from_pretrained(params['bert_model'])
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = ctxt_bert.config

    def forward(
            self,
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt,
            token_idx_cands,
            segment_idx_cands,
            mask_cands,
    ):   # 均为(B, 16,128)
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands  # 两个都是(B,768)


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path)

        self.model = self.model.to(self.device)
        # self.model = nn.DataParallel(self.model)
        # self.model = self.model.cuda()
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict)

    def build_model(self):
        self.model = BiEncoderModule(self.params)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    def encode_candidate(self, cands):   # 在使用检索前对每个候选实体编码
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX, 128
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def score_candidate(self, context_input, candidate_input, cand_encs=None):
        if cand_encs is not None:
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
                context_input, self.NULL_IDX, 128  # context_input:(B=1,128)
            )
            embedding_ctxt, _ = self.model(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
            )
            return embedding_ctxt.mm(cand_encs.t())

        else:
            # 进入bert格式之前需要先调整(这里context_input与candidate_input的size相同，调整方法一致)
            B = context_input.size(0)
            context_input = context_input.view(-1, context_input.size(-1))  # (B*16,128)
            candidate_input = candidate_input.view(-1, candidate_input.size(-1))  # (B*16,128)

            # context编码
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
                context_input, self.NULL_IDX, 128
            )  # 三个变量都是(B*16,128)

            # candidate编码
            token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
                candidate_input, self.NULL_IDX, 128
            )  # 三个变量都是(B*16,128)

            embedding_ctxt, embedding_cands = self.model(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt, token_idx_cands, segment_idx_cands, mask_cands
            )  # 两个变量都是(B*16, 768)

            # 还原shape:
            embedding_ctxt = embedding_ctxt.unsqueeze(1)   # (B*16, 1, 768)
            embedding_cands = embedding_cands.unsqueeze(2)  # (B*16, 768, 1)

            scores = torch.bmm(embedding_ctxt, embedding_cands)  # (B*16, 1, 1)
            scores = torch.squeeze(scores)  # (B*16,)
            scores = scores.view(B, -1)   # (B, 16)
            return scores

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(self, context_input, candidate_input, label_input):
        """
        context_input: (batch_size, 16, 128) 16个均为重复同一context
        candidate_input: (batch_size, 16, 128)
        label_input: (batch_size,)
        """
        scores = self.score_candidate(context_input, candidate_input)  # (B,16)
        # hard negatives,每个batch中只有一个label
        loss = F.cross_entropy(scores, label_input)
        return loss, scores


def to_bert_input(token_idx, null_idx, segment_pos):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    # generate segment_ids
    segment_idx = token_idx * 0
    if segment_pos > 0:
        segment_idx[:, segment_pos:] = token_idx[:, segment_pos:] > 0

    # generate attention_mask
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    # token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask   # out_size:(B, top_k, 128)
