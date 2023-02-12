# -*- coding: utf-8 -*-
# Date        : 2022/4/20
# Author      : Chen Xuekai
# Description :
import argparse
import json
import logging
import os
import torch
import random
import time
import numpy as np
from tqdm import tqdm, trange

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from blink.biencoder.hard_biencoder import BiEncoderRanker
import blink.biencoder.data_process as data
import blink.biencoder.nn_prediction as nnquery
from blink.biencoder.biencoder import load_biencoder
import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import WORLDS, load_entity_dict_zeshel, Stats  # 加载entity表
from blink.common.params import BlinkParser
from blink.biencoder.eval_biencoder import get_candidate_pool_tensor_zeshel, \
    load_or_generate_candidate_pool, encode_candidate

from blink.biencoder.train_biencoder import get_optimizer, get_scheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def evaluate(
        reranker, eval_dataloader, params, device, logger,
):
    reranker.model.eval()
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    acc = {}
    tot = {}
    world_size = len(WORLDS)
    for i in range(world_size):
        acc[i] = 0.0
        tot[i] = 0.0

    all_logits = []
    cnt = 0
    for step, batch in enumerate(iter_):
        cnt += 1
        batch = tuple(t.to(device) for t in batch)
        context_input, candidate_input, label_input, src = batch
        with torch.no_grad():
            eval_loss, logits = reranker(context_input, candidate_input, label_input)

        logits = logits.detach().cpu().numpy()
        label_ids = label_input.cpu().numpy()

        tmp_eval_accuracy, eval_result = utils.accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        all_logits.extend(logits)

        nb_eval_examples += context_input.size(0)
        for i in range(context_input.size(0)):
            src_w = src[i].item()
            acc[src_w] += eval_result[i]
            tot[src_w] += 1
        nb_eval_steps += 1

    normalized_eval_accuracy = -1
    if nb_eval_examples > 0:
        normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    macro = 0.0
    num = 0.0
    for i in range(len(WORLDS)):
        if acc[i] > 0:
            acc[i] /= tot[i]
            macro += acc[i]
            num += 1
    if num > 0:
        logger.info("Macro accuracy: %.5f" % (macro / num))
        logger.info("Micro accuracy: %.5f" % normalized_eval_accuracy)

    results["normalized_accuracy"] = normalized_eval_accuracy
    results["logits"] = all_logits
    return results


def main(params):
    # 日志存放位置
    logger = utils.get_logger(params["output_path"])

    # 模型初始化，path_to_model有值，直接加载已有模型
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model
    device = reranker.device
    n_gpu = reranker.n_gpu

    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # 分别用hard_bi和cross对top16重排序，比较准确率
    # 预处理：context_input堆叠
    fname = os.path.join(params["data_path"], "test.t7")
    test_data = torch.load(fname)  # {context_vecs,candidate_vecs,labels,candidate_id,worlds:}
    context_input = test_data["context_vecs"]  # [6669, 128]
    candidate_input = test_data["candidate_vecs"]  # [6669, 16, 128]
    label_input = test_data["labels"]  # [6669,]
    src_input = test_data['worlds'][:len(context_input)]  # [6669,] world index
    # 每条context_input堆16次
    repeat_context_input = torch.repeat_interleave(context_input, 16, dim=0)
    repeat_context_input = repeat_context_input.view(-1, 16, 128)  # [6669, 16, 128]

    test_tensor_data = TensorDataset(repeat_context_input, candidate_input, label_input, src_input)
    test_sampler = SequentialSampler(test_tensor_data)  # 由原来的random采样改为顺序采样，保证每16个组内负例不变
    test_dataloader = DataLoader(
        test_tensor_data,
        sampler=test_sampler,
        batch_size=params["eval_batch_size"]
    )

    # 开始测试
    time_start = time.time()
    logger.info("Starting testing")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )
    results = evaluate(
        reranker,
        test_dataloader,
        params=params,
        device=device,
        logger=logger
    )
    execution_time = (time.time() - time_start) / 60
    logger.info("The testing took {} minutes\n".format(execution_time))


if __name__ == '__main__':
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()
    args = parser.parse_args()
    params = args.__dict__
    main(params)


