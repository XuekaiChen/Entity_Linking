# -*- coding: utf-8 -*-
# Date        : 2022/3/31
# Author      : Chen Xuekai
# Description : 7666*(1-0.72) badcase

import ast
import torch
import jsonlines
import json
import pandas as pd
import os
from blink.biencoder.zeshel_utils import WORLDS
from extract_IR_badcase import load_entity_dict


if __name__ == '__main__':
    bad_id_path = "blink_cross_bad_id_list.txt"
    with open(bad_id_path, 'r', encoding='utf-8') as f:
        badcase_id = ast.literal_eval(f.read())
    print("badcase number: ",len(badcase_id))

    bad_id_path = "blink_cross_bad_goldrank_list.txt"
    with open(bad_id_path, 'r', encoding='utf-8') as f:
        gold_rank_from_cross = ast.literal_eval(f.read())  # list类型
    print("valid number from gold_rank_file: ", len(gold_rank_from_cross))
    print(gold_rank_from_cross[:10])

    rerank_candidates_id_path = "blink_cross_bad_rerank_candidate_id.txt"
    with open(rerank_candidates_id_path, 'r', encoding='utf-8') as f:
        rerank_candidates_id = ast.literal_eval(f.read())  # list:[2178,64]
    print("rerank_candidate_id number: ", len(rerank_candidates_id))

    fname = "models/zeshel/all_sample_top64_candidates/test.t7"
    test_data = torch.load(fname)
    context_input = test_data["context_vecs"]  # [10000, 128]
    candidate_input = test_data["candidate_vecs"]  # [10000, 64, 128]
    label_input = test_data["labels"].tolist()  # [10000] 位次
    candidate_ids = test_data['candidate_id'].tolist()  # [10000, 64]
    world_id_list = test_data['worlds'].tolist()
    badcase_world = []
    bad_candidate_ids = []  # [2178,64]
    gold_rank_from_bi = []
    for bad_id in badcase_id:
        bad_candidate_ids.append(candidate_ids[bad_id])
        badcase_world.append(world_id_list[bad_id])
        gold_rank_from_bi.append(label_input[bad_id])  # 但这个位次是biencoder检索的位次

    # 加载entity{label:id}，候选实体转为单词
    # entity_dic: (16 world, entity{title,text,document_id})
    # entity_map: (16 world, {document_id: label_id})
    entity_dic, entity_map = load_entity_dict()
    entity_id2title = {}
    context_list = []
    target_list = []
    birank_candidates = []
    crossrank_candidates = []
    world_list = []

    with open("data/zeshel/blink_format/test.jsonl", 'r', encoding="utf-8") as f:
        pointer = 0
        for idx, line in enumerate(jsonlines.Reader(f)):
            if idx == badcase_id[pointer]:
                context = line['context_left']+"### "+line['mention']+" ###"+line['context_right']
                context_list.append(context)
                target_list.append(line['label_title'])
                src = WORLDS[badcase_world[pointer]]
                world_list.append(src)

                # 按bi_encoder的排名列出实体
                candidates = []
                for label_id in bad_candidate_ids[pointer]:
                    candidates.append(entity_dic[src][label_id]['title'])
                birank_candidates.append(candidates)

                # 按cross_encoder的排名列出实体
                candidates = []
                for label_id in rerank_candidates_id[pointer]:
                    candidates.append(entity_dic[src][label_id]['title'])
                crossrank_candidates.append(candidates)

                # 结束遍历2178个负例出口
                pointer += 1
                if pointer == len(badcase_id):
                    break

    df = pd.DataFrame()
    df['sentence'] = context_list
    df['world'] = world_list
    df['target'] = target_list
    # 可以根据以下四项找到hard negatives
    df['bi_gold_rank'] = gold_rank_from_bi
    df['cross_gold_rank'] = gold_rank_from_cross
    df['bi_rank_candidates'] = birank_candidates
    df['cross_rank_candidates'] = crossrank_candidates
    df.to_csv("blink_cross_bad_case.csv", index=False)
    print("finish writing badcase to blink_cross_bad_case.csv")
