import blink.main_dense as main_dense
import argparse
import jsonlines

import json

models_path = "models/" # the path where you stored the BLINK models

config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 10,
    "biencoder_model": models_path+"biencoder_wiki_large.bin",
    "biencoder_config": models_path+"biencoder_wiki_large.json",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
    "crossencoder_config": models_path+"crossencoder_wiki_large.json",
    "fast": True, # set this to be true if speed is a concern
    "output_path": "logs/" # logging directory
}

args = argparse.Namespace(**config)

models = main_dense.load_models(args, logger=None)


data_path = "ace2004-test-kilt.jsonl"
data_to_link_acetry = []
with open(data_path, 'r') as f:
    count = 0
    for line in jsonlines.Reader(f):
        line_dict = {"id": count, "label": "unknow", "label_id": -1, "context_left": line["meta"]["left_context"],
                     "mention": line["meta"]["mention"], "context_right": line["meta"]["right_context"]}
        count += 1
        data_to_link_acetry.append(line_dict)

# 数据格式
data_to_link = [{
                    "id": 0,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "".lower(),
                    "mention": "Shakespeare".lower(),
                    "context_right": "'s account of the Roman general Julius Caesar's murder by his friend Brutus is a meditation on duty.".lower(),
                },
                {
                    "id": 1,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "Shakespeare's account of the Roman general".lower(),
                    "mention": "Julius Caesar".lower(),
                    "context_right": "'s murder by his friend Brutus is a meditation on duty.".lower(),
                }
                ]

_, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link_acetry)

prediction_path = "ace2004_test_prediction.jsonl"
for prediction in predictions:
    entry = {"prediction": prediction}
    with open(prediction_path, "a") as f:
        json.dump(entry, f)
        f.write("\n")



