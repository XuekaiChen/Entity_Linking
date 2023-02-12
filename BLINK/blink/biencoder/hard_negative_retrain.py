# -*- coding: utf-8 -*-
# Date        : 2022/4/8
# Author      : Chen Xuekai
# Description : 改编eval_biencoder，检索top15_hard_negative，自写train过程

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
import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import WORLDS, load_entity_dict_zeshel, Stats  # 加载entity表
from blink.common.params import BlinkParser
from blink.biencoder.eval_biencoder import get_candidate_pool_tensor_zeshel, \
    load_or_generate_candidate_pool, encode_candidate

from blink.biencoder.train_biencoder import get_optimizer, get_scheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
"""
待输入参数：
--path_to_model models/zeshel/biencoder/pytorch_model.bin 
--data_path models/zeshel/hard_biencoder
--cand_encode_path models/zeshel/train_candidate_pool.t7
--cand_pool_path models/zeshel/train_cand_encode.t7
--output_path models/zeshel/hard_biencoder
--max_context_length 128
--max_cand_length 128
--top_k 16 
--save_topk_result 
--bert_model bert-large-uncased 
--mode train,valid,test
--zeshel True 
--type_optimization  all_encoder_layers
--data_parallel
"""
# 首先将train/valid/test的cand_encode存起来--√
# 再写程序进行操作
"""
--learning_rate 1e-05
--num_train_epochs 5
--train_batch_size 8
--eval_batch_size 8
"""


def main(params):
    # 日志存放位置
    logger = utils.get_logger(params["output_path"])
    output_path = params["output_path"]

    # 模型初始化，path_to_model有值，直接加载已有模型
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model
    device = reranker.device
    n_gpu = reranker.n_gpu

    # 加载cand_pool和cand_encode（起初两个都没有，生成后屏蔽）
    # cand_pool
    # cand_pool_path = params.get("cand_pool_path", None)
    #
    # candidate_pool = load_or_generate_candidate_pool(tokenizer,
    #                                                  params,
    #                                                  logger,
    #                                                  cand_pool_path)
    # cand_encode_path = params.get(params['mode'] + "_cand_encode_path", None)
    # # cand_encode
    # candidate_encoding = encode_candidate(
    #     reranker,
    #     candidate_pool,
    #     params["encode_batch_size"],
    #     silent=params["silent"],
    #     logger=logger,
    #     is_zeshel=params.get("zeshel", None)
    # )
    # # Save candidate encoding to avoid re-compute
    # logger.info("Saving candidate encoding to file " + cand_encode_path)
    # torch.save(candidate_encoding, cand_encode_path)

    # 仿写train_cross.py
    # 假设已经加载好cand_encode（屏蔽之前代码），直接读取train/valid/test.t7
    # TODO:需要用训练好的模型给valid_cand编码

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    max_seq_length = params["max_cand_length"]
    context_length = params["max_context_length"]
    eval_batch_size = params['eval_batch_size']

    # 修正训练集to: train_dataloader（直接加载16的
    fname = os.path.join(params["data_path"], "train.t7")
    train_data = torch.load(fname)  # {context_vecs,candidate_vecs,labels,candidate_id,worlds:}
    context_input = train_data["context_vecs"]  # [49275, 128]
    candidate_input = train_data["candidate_vecs"]  # [49275, 16, 128]
    label_input = train_data["labels"]  # [49275,]
    src_input = train_data['worlds'][:len(context_input)]  # [49275,] world index

    # 每条context_input堆16次
    repeat_context_input = torch.repeat_interleave(context_input, 16, dim=0)
    repeat_context_input = repeat_context_input.view(-1, 16, 128)  # [43678, 16, 128]

    train_tensor_data = TensorDataset(repeat_context_input, candidate_input, label_input, src_input)
    train_sampler = SequentialSampler(train_tensor_data)  # 由原来的random采样改为顺序采样，保证每16个组内负例不变
    train_dataloader = DataLoader(
        train_tensor_data,
        sampler=train_sampler,
        batch_size=params["train_batch_size"]
    )

    # 重新读取valid_sample
    # valid_samples = utils.read_dataset("valid", "data/zeshel/blink_format")
    # logger.info("Read %d valid samples." % len(valid_samples))
    #
    # valid_data, valid_tensor_data = data.process_mention_data(
    #     valid_samples,
    #     tokenizer,
    #     params["max_context_length"],
    #     params["max_cand_length"],
    #     context_key=params["context_key"],
    #     silent=params["silent"],
    #     logger=logger,
    #     debug=params["debug"],
    # )  # valid_tensor_data:{context_vecs,cand_vecs,label_idx,src}
    # valid_sampler = SequentialSampler(valid_tensor_data)
    # valid_dataloader = DataLoader(
    #     valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    # )

    # 训练之前先测一下原bi-encoder效果，看看hard_negative能不能成功
    # TODO：这个Eval accuracy为什么是一个接近1的数？？
    # results = evaluate(
    #     reranker,
    #     valid_dataloader,
    #     params=params,
    #     device=device,
    #     logger=logger
    # )

    # 开始训练，由于train数据是biencoder分好的，所以仿写train_cross训练过程
    time_start = time.time()
    utils.write_to_file(
        os.path.join(params["output_path"], "training_params.txt"), str(params)
    )
    logger.info("----------Starting training----------")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )
    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)
    model.train()
    best_epoch_idx = -1
    best_score = -1
    num_train_epochs = params["num_train_epochs"]

    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):  # 5轮
        tr_loss = 0
        results = None
        iter_ = tqdm(train_dataloader, desc="Batch")
        part = 0
        # 改为一个batch内多组，对比学习
        for step, batch in enumerate(iter_):  # 每个batch
            batch = tuple(t.to(device) for t in batch)
            context_input, candidate_input, label_input, _ = batch  # [4,16,128],[4,16,128],[4,]
            # TODO：还一个关键问题，data_process里面的select_field，即不同world怎么办？
            loss, _ = reranker(context_input, candidate_input, label_input)
            tr_loss += loss.item()
            if (step + 1) % params["print_interval"] == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / params["print_interval"],
                    )
                )
                tr_loss = 0
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # 去除evaluate环节，因为测试需要单独用模型编码+检索，不可在batch内做分类
            # if (step + 1) % params["eval_interval"] == 0:
            #         logger.info("Evaluation on the development dataset")
            #         evaluate(
            #             reranker,
            #             valid_dataloader,
            #             params=params,
            #             device=device,
            #             logger=logger
            #         )
            #     logger.info("***** Saving fine - tuned model *****")
            #     epoch_output_folder_path = os.path.join(
            #         params['output_path'], "epoch_{}_{}".format(epoch_idx, part)
            #     )
            #     part += 1
            #     utils.save_model(model, tokenizer, epoch_output_folder_path)
            #     model.train()
            #     logger.info("\n")
        # 一轮结束
        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            params['output_path'], "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path)
        # reranker.save(epoch_output_folder_path)

        # output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        # results = evaluate(
        #     reranker,
        #     valid_dataloader,
        #     params=params,
        #     device=device,
        #     logger=logger
        # )
        #
        # ls = [best_score, results["normalized_accuracy"]]
        # li = [best_epoch_idx, epoch_idx]
        #
        # best_score = ls[np.argmax(ls)]
        # best_epoch_idx = li[np.argmax(ls)]
        # logger.info("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(params['output_path'], "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    # logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    # params["path_to_model"] = os.path.join(
    #     params['output_path'], "epoch_{}".format(best_epoch_idx)
    # )


if __name__ == "__main__":
    # 添加model、train、eval三个参数群
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()
    args = parser.parse_args()
    print(args)
    params = args.__dict__
    main(params)

    # # 对train/valid/test三个集合都做pool/encode/evaluate
    # mode_list = params["mode"].split(',')
    # for mode in mode_list:
    #     new_params = params
    #     new_params["mode"] = mode
    #     new_params['cand_pool_path'] = params['cand_pool_path'] + mode + "_candidate_pool.t7"
    #     new_params['cand_encode_path'] = params['cand_encode_path'] + mode + "_cand_encode.t7"
    #     main(new_params)
