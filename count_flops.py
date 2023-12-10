import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

import torch
from torch import nn
from transformers import BertTokenizer
import re
# from transformers import AutoTokenizer

import config as CFG
from dataset import CLIPDataset, get_transforms
from CLIP import CLIPModel
from utils import AvgMeter, get_lr
import argparse

import torch.nn.functional as F
from matplotlib import pyplot as plt
import cv2
from torch.profiler import profile, record_function, ProfilerActivity

# for i in range(144):
parser = argparse.ArgumentParser(description='Your program description')
# parser.add_argument('--start_subnet', type=int, help='Choose the subnet no to start testing from')
parser.add_argument('--sampling_function', type=str, choices=['randomized_sampling', 'big_small_sampling', 'no_sampling', 'supernet_subnet_sampling'], default='random',
                    help='Choose the sampling function for subnets (random or your custom function)')

args = parser.parse_args()

def get_flops(valid_df, model_path, subnet=0):
    tokenizer = BertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.change_image_encoder_subnet(int(subnet/12))
    model.change_text_encoder_subnet(int(subnet%12))
    print(sum(p.numel() for p in model.parameters()))
    model.eval()
    
    valid_image_embeddings = []
    dot_similarity = []
    flops = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            with profile(
                activities=[ProfilerActivity.CUDA],
                with_stack=True,
            ) as prof:
                image_features = model.image_encoder(batch["image"].to(CFG.device))
                image_embeddings = model.image_projection(image_features)


                text_features = model.text_encoder(
                    input_ids=batch["input_ids"].to(CFG.device), attention_mask=batch["attention_mask"].to(CFG.device)
                )[1]
                text_embeddings = model.text_projection(text_features)
            
            pattern = re.compile(r"Self CUDA time total: (\d+\.\d+)ms")
            flops = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)

            # Find the match in the string
            match = re.search(pattern, flops)

            # Extract the float value
            if match:
                float_value = float(match.group(1))
                print(f"The float value is: {float_value}")
            else:
                print("No match found.")
            # flops = sum([item.cuda_time for item in prof.key_averages()])/1000.00
            # flops = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
            # print(f'Total FLOPs: {flops:.2f}')
            # print(float_value)
            # break
            return float_value

            # image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
            # text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)

            # similarity = text_embeddings_n @ image_embeddings_n.T
            # print(similarity.shape)
            # dot_similarity.append((text_embeddings_n @ image_embeddings_n.T)[0][0])
            # valid_image_embeddings.append(image_embeddings)
    return flops

def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{CFG.captions_path}/captions.csv")
    max_id = dataframe.index.max() + 1 if not CFG.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe.index.isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe.index.isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


def find_matches(model, image_embeddings, query, image_filenames, n=9, subnet=0):
    tokenizer = BertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        # print("Subnet:", subnet)
        
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )[1]
        # print(model.text_encoder.model.encoder)
        # print(model.text_encoder.model.encoder.width_mult)
        # print(text_features)
        text_embeddings = model.text_projection(text_features)
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]
    
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{CFG.image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")
    
    plt.savefig(f'Best_ResnetDynabert_{args.sampling_function}_{subnet}.png')

save_flops = np.zeros(108)

for subnet in range(108):
    # subnet = 0
    _, valid_df = make_train_valid_dfs()
    flops = get_flops(valid_df, f'Best_ResnetDynabert_{args.sampling_function}.pt', subnet=subnet)
    # print(flops)
    save_flops[subnet] = flops
    # print(save_dot_similarity_matrix[subnet - args.start_subnet])

np.save(f'{args.sampling_function}_flops.npy', save_flops)
# print(dot_similarity)
# find_matches(model, 
#              image_embeddings,
#              query="a boy jumping with a skateboard",
#              image_filenames=valid_df['image'].values,
#              n=9,
#              subnet=subnet)