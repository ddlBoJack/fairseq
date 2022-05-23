import torch
import fairseq
import json
import logging
import os
import sys


# model_raw = torch.load("/home/v-ziyangma/model/data2vec/download_pretrained/audio_base_ls.pt")
model_raw = torch.load("/home/v-ziyangma/model/data2vec_uni/pretrain_debug/checkpoint_1_1.pt")
# cfg = model_raw['cfg']
model = model_raw['model']

# compute parameters
sum_param = 0
for k, v in model.items():
    if k!='_ema':
        sum_param += v.numel()
print(sum_param)

# with open("/home/v-ziyangma/config/ours/data2vec.json", 'w') as f:
#     f.write(json.dumps(cfg, indent=4, sort_keys=True))