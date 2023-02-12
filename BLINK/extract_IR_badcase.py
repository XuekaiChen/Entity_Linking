# -*- coding: utf-8 -*-
# Date        : 2022/3/25
# Author      : Chen Xuekai
# Description : 10000-7666=2334 badcase

import torch
import jsonlines
import json
import pandas as pd
import os
from blink.biencoder.zeshel_utils import WORLDS


def load_entity_dict():
    entity_dict = {}
    entity_map = {}
    for src in WORLDS:
        fname = os.path.join("data/zeshel/documents", src + ".json")
        assert os.path.isfile(fname), "File not found! %s" % fname
        cur_dict = {}
        doc_map = {}
        doc_list = []
        with open(fname, 'rt') as f:
            for line in f:
                line = line.rstrip()
                item = json.loads(line)
                doc_id = item["document_id"]
                title = item["title"]
                text = item["text"]
                doc_map[doc_id] = len(doc_list)
                doc_list.append(item)

        print("Load for world %s." % src)
        entity_dict[src] = doc_list
        entity_map[src] = doc_map

    return entity_dict, entity_map


if __name__ == '__main__':
    fname = "models/zeshel/all_sample_top64_candidates/test.t7"
    test_data = torch.load(fname)
    context_input = test_data["context_vecs"]  # [10000, 128]
    candidate_input = test_data["candidate_vecs"]  # [10000, 64, 128]
    label_input = test_data["labels"].tolist()  # [10000]
    candidate_ids = test_data['candidate_id'].tolist()  # [10000, 64]
    world_id_list = test_data['worlds'].tolist()

    badcase_id = []
    badcase_world = []
    bad_candidate_ids = []
    for case_id, case in enumerate(label_input):
        if case == -1:  # 64个中无检索
            badcase_id.append(case_id)
            bad_candidate_ids.append(candidate_ids[case_id])
            badcase_world.append(world_id_list[case_id])

    # 加载entity{label:id}，候选实体转为单词
    # entity_dic: (16 world, entity{title,text,document_id})
    # entity_map: (16 world, {document_id: label_id})
    entity_dic, entity_map = load_entity_dict()
    entity_id2title = {}
    context_list = []
    target_list = []
    candidates_list = []
    world_list = []

    with open("data/zeshel/blink_format/test.jsonl", 'r', encoding="utf-8") as f:
        pointer = 0
        for idx, line in enumerate(jsonlines.Reader(f)):
            if idx == badcase_id[pointer]:
                context = line['context_left']+"### "+line['mention']+" ###"+line['context_right']
                context_list.append(context)
                target_list.append(line['label_title'])
                candidates = []
                src = WORLDS[badcase_world[pointer]]
                world_list.append(src)
                for label_id in bad_candidate_ids[pointer]:
                    candidates.append(entity_dic[src][label_id]['title'])
                candidates_list.append(candidates)
                pointer += 1
                if pointer == len(badcase_id):
                    break

    df = pd.DataFrame()
    df['sentence'] = context_list
    df['world'] = world_list
    df['target'] = target_list
    df['candidates'] = candidates_list
    df.to_csv("blink_bi_bad_case.csv", index=False)
