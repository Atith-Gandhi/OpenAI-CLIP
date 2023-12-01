import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

import torch
from torch import nn
from transformers import BertTokenizer
# from transformers import AutoTokenizer

import config as CFG
from dataset import CLIPDataset, get_transforms
from CLIP import CLIPModel
from utils import AvgMeter, get_lr
import argparse

import torch.nn.functional as F
from matplotlib import pyplot as plt
import cv2

# for i in range(144):

def get_dot_similarity(valid_df, model_path, subnet=0):
    tokenizer = BertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.change_image_encoder_subnet(int(subnet/12))
    model.change_text_encoder_subnet(int(subnet%9))
    model.eval()
    
    valid_image_embeddings = []
    dot_similarity = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)

            text_features = model.text_encoder(
                input_ids=batch["input_ids"].to(CFG.device), attention_mask=batch["attention_mask"].to(CFG.device)
            )[1]
            text_embeddings = model.text_projection(text_features)

            image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
            text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)

            similarity = text_embeddings_n @ image_embeddings_n.T
            print(similarity.shape)
            dot_similarity.append(text_embeddings_n @ image_embeddings_n.T)
            # valid_image_embeddings.append(image_embeddings)
    return dot_similarity

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
        print("Subnet:", subnet)
        
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )[1]
        # print(model.text_encoder.model.encoder)
        # print(model.text_encoder.model.encoder.width_mult)
        print(text_features)
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
    
    plt.savefig(f'Best_ResnetDynabert_randomized_sampling_{subnet}.png')

subnet = 0
_, valid_df = make_train_valid_dfs()
dot_similarity = get_dot_similarity(valid_df, "Best_ResnetDynabert_randomized_sampling.pt", subnet=subnet)

print(dot_similarity)
# find_matches(model, 
#              image_embeddings,
#              query="a boy jumping with a skateboard",
#              image_filenames=valid_df['image'].values,
#              n=9,
#              subnet=subnet)