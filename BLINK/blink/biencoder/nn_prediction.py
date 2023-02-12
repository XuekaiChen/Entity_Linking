# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import logging
import torch
from tqdm import tqdm

import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import WORLDS, Stats


def get_topk_predictions(
    reranker,
    train_dataloader,
    candidate_pool,
    cand_encode_list,
    silent,
    logger,
    top_k=10,
    is_zeshel=False,
    save_predictions=False,
):
    reranker.model.eval()
    device = reranker.device
    logger.info("Getting top %d predictions." % top_k)
    if silent:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader)

    nn_context = []
    nn_candidates = []
    nn_labels = []
    nn_worlds = []
    nn_topk_candidate_id = torch.tensor([])
    stats = {}

    if is_zeshel:
        world_size = len(WORLDS)
    else:
        # only one domain
        world_size = 1
        candidate_pool = [candidate_pool]
        cand_encode_list = [cand_encode_list]

    logger.info("World size : %d" % world_size)

    for i in range(world_size):
        stats[i] = Stats(top_k)  # 每个world的top_k recall
    
    oid = 0
    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, _, m_position, srcs, label_ids = batch
        src = srcs[0].item()
        scores = reranker.score_candidate(
            context_input, 
            None,
            m_position=m_position,
            cand_encs=cand_encode_list[src].to(device),  #在domain全词表中检索
        )
        # indicies存储topkrecall entity在cand_encode_list中的编号
        values, indicies = scores.topk(top_k)
        old_src = src
        for i in range(context_input.size(0)):  # 遍历每个样本
            oid += 1
            inds = indicies[i]  # batch中样本i对应的top_k entity编号

            if srcs[i] != old_src:
                src = srcs[i].item()
                # not the same domain, need to re-do
                new_scores = reranker.score_candidate(
                    context_input[[i]], 
                    None,
                    m_position=m_position,
                    cand_encs=cand_encode_list[src].to(device),
                )
                _, inds = new_scores.topk(top_k)
                inds = inds[0]

            pointer = -1
            for j in range(top_k):  # 只看前top_k个，若存在gold，则pointer为gold序号；若不存在，则认为-1
                if inds[j].item() == label_ids[i].item():   # inds为tensor
                    pointer = j
                    break
            stats[src].add(pointer)

            # 若没在前top_k，则pointer=-1，不添加到nn_candidate中
            # if pointer == -1:
            #     continue  # 现在无论在不在64个中，都将返回64个候选实体id，并转换成label

            # hard_retrain支线：若不存在，则替换第15个位gold entity编号
            # if pointer == -1:
            #     inds[top_k-1] = label_ids[i].item()  # 看眼什么数据类型，有可能是int
            #     pointer = top_k-1

            if not save_predictions:
                continue

            # add examples in new_data
            # test_candidate_pool: (world_num, samples_num, embed_dim)-->(4,15603,128)
            # 针对单个context
            cur_candidates = candidate_pool[src][inds]  # (16,128)
            nn_context.append(context_input[i].cpu().tolist())
            nn_candidates.append(cur_candidates.cpu().tolist())
            nn_labels.append(pointer)
            nn_worlds.append(src)
            # 可以在这里返回inds，即batch中一个样本的64个candidate_id
            nn_topk_candidate_id = torch.cat([nn_topk_candidate_id, inds.cpu().unsqueeze(0)], dim=0)

    res = Stats(top_k)
    for src in range(world_size):
        if stats[src].cnt == 0:
            continue
        if is_zeshel:
            logger.info("In world " + WORLDS[src])
        output = stats[src].output()
        logger.info(output)
        res.extend(stats[src])

    logger.info(res.output())

    nn_context = torch.LongTensor(nn_context)
    nn_candidates = torch.LongTensor(nn_candidates)
    nn_labels = torch.LongTensor(nn_labels)
    nn_topk_candidate_id = torch.LongTensor(nn_topk_candidate_id.numpy())
    nn_data = {
        'context_vecs': nn_context,  # (10000,128)
        'candidate_vecs': nn_candidates,  # (10000,top_k,128)
        'labels': nn_labels,  # (10000,) gold_entity在candidate中的位次，top_k中没有则返回-1
        'candidate_id': nn_topk_candidate_id  # (10000,top_k) 所有候选实体的id
    }

    if is_zeshel:
        nn_data["worlds"] = torch.LongTensor(nn_worlds)
    
    return nn_data

