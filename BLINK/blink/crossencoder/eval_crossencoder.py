# -*- coding: utf-8 -*-
# Date        : 2022/3/10
# Author      : Chen Xuekai
# Description :
import torch
import os
import random
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from blink.crossencoder.crossencoder import load_crossencoder
from blink.biencoder.biencoder import load_biencoder
import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS
from blink.common.params import BlinkParser

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# concatenate the context and each entity description (1-64 -> 64-64)
def modify(context_input, candidate_input, max_seq_length):
    new_input = []
    context_input = context_input.tolist()
    candidate_input = candidate_input.tolist()

    for i in range(len(context_input)):
        cur_input = context_input[i]
        cur_candidate = candidate_input[i]
        mod_input = []
        for j in range(len(cur_candidate)):  # concatenate context with each candidate description
            sample = cur_input + cur_candidate[j][1:]  # remove [CLS] token from candidate
            sample = sample[:max_seq_length]
            mod_input.append(sample)

        new_input.append(mod_input)

    return torch.LongTensor(new_input)


def evaluate(reranker, test_dataloader, device, logger, context_length):
    reranker.model.eval()
    iter_ = tqdm(test_dataloader, desc="Test")

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

    all_scores = []
    cnt = 0
    bad_id = []
    gold_rank = []
    rerank_candidates_id = []  # [7666,64]
    for step, batch in enumerate(iter_):
        # batch:{context_input,label_input,src,sentence_id_in_10000,candidates_id}
        src = batch[2]  # world
        sentence_id = batch[3].tolist()  # [B,]
        cnt += 1
        batch = tuple(t.to(device) for t in batch)
        context_input = batch[0]  # [B,64,128]
        label_input = batch[1]
        candidates_id = batch[4].tolist()  # [B,64]
        with torch.no_grad():
            eval_loss, scores = reranker(context_input, label_input, context_length)

        scores = scores.detach().cpu().numpy()  # [B, 64]
        label_ids = label_input.cpu().numpy()  # [B,]

        # 相同的个数、true/false数组[B,]
        tmp_eval_accuracy, eval_result = utils.accuracy(scores, label_ids)

        eval_accuracy += tmp_eval_accuracy
        all_scores.extend(scores)

        scores = scores.tolist()
        label_ids = label_ids.tolist()
        for idx, sample in enumerate(scores):  # 遍历batch内每个样本
            # 对score进行排序，返回index列表
            rank_list = sorted(range(len(sample)), key=lambda x: sample[x], reverse=True)  # [64,]
            rank = rank_list.index(label_ids[idx])  # gold的位次
            if rank != 0:  # 如果是badcase
                gold_rank.append(rank)
                bad_id.append(sentence_id[idx])
                # 找rank位次前词的编号，编号转candidate_id
                rerank_candidates_id.append([candidates_id[idx][cand] for cand in rank_list])

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
    results["scores"] = all_scores
    return results, bad_id, gold_rank, rerank_candidates_id


def main(params):
    logger = utils.get_logger(params["output_path"])

    # load cross-encoder model
    # reranker = load_crossencoder(params)
    reranker = load_biencoder(params)  # TODO：看看使用bi-encoder重排序的效果
    device = reranker.device
    n_gpu = reranker.n_gpu

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    max_seq_length = params['max_seq_length']
    context_length = params['max_context_length']

    # load test data from all_sample_top64_candidates
    fname = os.path.join(params["data_path"], "test.t7")
    test_data = torch.load(fname)

    # 提取有召回的样本和id; 10000-->7666
    context_input = []
    candidate_input = []
    label_input = []
    all_context_input = test_data['context_vecs'].tolist()  # list:[10000,128]
    all_candidate_input = test_data['candidate_vecs'].tolist()  # list:[10000,64,128]
    all_label_input = test_data['labels'].tolist()  # list:[10000]
    all_candidate_id = test_data['candidate_id'].tolist()  # list:[10000,64]
    recall_sample_id = []  # [7666]
    recall_candidates_id = []  # [7666,64]
    for idx, label in enumerate(all_label_input):
        if label != -1:  # 有召回样本
            context_input.append(all_context_input[idx])
            candidate_input.append(all_candidate_input[idx])
            label_input.append(all_label_input[idx])
            recall_sample_id.append(idx)
            recall_candidates_id.append(all_candidate_id[idx])
    del all_context_input, all_candidate_input, all_label_input, all_candidate_id
    context_input = torch.LongTensor(context_input)
    candidate_input = torch.LongTensor(candidate_input)
    label_input = torch.LongTensor(label_input)
    recall_sample_id = torch.LongTensor(recall_sample_id)
    recall_candidates_id = torch.LongTensor(recall_candidates_id)
    context_input = modify(context_input, candidate_input, max_seq_length)  # [7666,64,128]
    src_input = test_data["worlds"][:len(context_input)]
    test_tensor_data = TensorDataset(context_input, label_input, src_input, recall_sample_id, recall_candidates_id)
    test_sampler = SequentialSampler(test_tensor_data)

    test_dataloader = DataLoader(
        test_tensor_data,
        sampler=test_sampler,
        batch_size=params["eval_batch_size"]
    )

    time_start = time.time()
    logger.info("Starting testing")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )
    results, bad_id, gold_rank, rerank_candidates_id = evaluate(   # bad_id:list
        reranker,
        test_dataloader,
        device=device,
        logger=logger,
        context_length=context_length
    )
    execution_time = (time.time() - time_start) / 60
    logger.info("The testing took {} minutes\n".format(execution_time))

    # badcase_filename = "blink_cross_bad_id_list.txt"
    # logger.info("Writing bad_id to {}".format(badcase_filename))
    # with open(badcase_filename, 'w') as f:
    #     f.write(str(bad_id))
    #
    # bad_goldrank_path = "blink_cross_bad_goldrank_list.txt"
    # logger.info("Writing gold_rank to {}".format(bad_goldrank_path))
    # with open(bad_goldrank_path, 'w') as f:
    #     f.write(str(gold_rank))
    #
    # bad_rerank_path = "blink_cross_bad_rerank_candidate_id.txt"
    # logger.info("Writing rerank_candidate_id to {}".format(bad_rerank_path))
    # with open(bad_rerank_path, 'w') as f:
    #     f.write(str(rerank_candidates_id))


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()
    args = parser.parse_args()
    params = args.__dict__
    main(params)
