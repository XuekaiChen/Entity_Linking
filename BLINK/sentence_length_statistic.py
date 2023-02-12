# -*- coding: utf-8 -*-
# Date        : 2022/3/14
# Author      : Chen Xuekai
# Description : statistic of entity descriptions, check whether it worth to split paragraph

from pytorch_transformers import BertTokenizer
import json
import matplotlib.pyplot as plt
import time


def get_context_ids(filename):
    contexts = []
    count = 0
    length_dict = {"0-128": 0, "128-256": 0, "256-512": 0, ">512": 0}
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            context_tokens = tokenizer.tokenize(line["text"])
            context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
            # context_ids = tokenizer.convert_tokens_to_ids(context_tokens)
            length = len(context_tokens)
            if length < 128:
                length_dict["0-128"] += 1
            elif 128 < length < 256:
                length_dict["128-256"] += 1
            elif 256 < length < 512:
                length_dict["256-512"] += 1
            else:
                length_dict[">512"] += 1
            contexts.append(context_tokens)
            count += 1
    return contexts, length_dict, count

WORLDS = [
    'fallout',
    'american_football',
    'doctor_who',
    'final_fantasy',
    'military',
    'pro_wrestling',
    'starwars',
    'world_of_warcraft',
    'coronation_street',
    'muppets',
    'ice_hockey',
    'elder_scrolls',
    'forgotten_realms',
    'lego',
    'star_trek',
    'yugioh'
]
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
start = time.time()
for filename in WORLDS:
    file_path = "data/zeshel/documents/{}.json".format(filename)
    context_ids, length_dict, count = get_context_ids(file_path)
    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    ax.set_title("{} sentence length".format(filename), fontsize=14)
    plt.bar(length_dict.keys(), length_dict.values(), color='r',width=0.5)
    for k,v in length_dict.items():
        plt.text(k, v, "%.2f%%" % (100*v/count), ha="center", va="bottom", fontsize=12)
    plt.savefig("{}.png".format(filename))
    time_cost = time.time() - start
    print("file %s cost %.2f second" % (filename, time_cost))

total_time_cost = time.time() - start
print("total time cost: %.2f minutes" % (total_time_cost/60))




