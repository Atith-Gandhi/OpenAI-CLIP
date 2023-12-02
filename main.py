import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

import torch
from torch import nn
from transformers import BertTokenizer
# from transformers import BertTokenizer

import config as CFG
from dataset import CLIPDataset, get_transforms
from CLIP import CLIPModel
from utils import AvgMeter, get_lr
import argparse
import itertools

# Define command-line arguments
parser = argparse.ArgumentParser(description='Your program description')
parser.add_argument('--sampling-function', type=str, choices=['randomized_sampling', 'big_small_sampling', 'no_sampling'], default='random',
                    help='Choose the sampling function for subnets (random or your custom function)')

args = parser.parse_args()

def get_no_sampling_subnets():
    return [0]

def get_random_subnets():
    sampled_subnets =  random.sample(range(108), 2)
    sampled_subnets.append(0)
    return sampled_subnets

def get_big_small_subnets():
    sampled_small_image_subnets = random.sample(range(4), 2)
    sampled_big_text_subnets = random.sample(range(6), 2)
    sampled_big_image_subnets = random.sample(range(5, 9), 2)
    sampled_small_text_subnets = random.sample(range(6, 12), 2)

    # print(sampled_small_image_subnets)
    list1 = list((np.array(sampled_small_image_subnets)*12 + np.array(sampled_big_text_subnets)).tolist())
    list2 = list((np.array(sampled_big_image_subnets)*12 + np.array(sampled_small_text_subnets)).tolist())
    # print(list1)
    # print(list2)
    list1.extend(list2)
    # print(list1)

    sampled_subnets = list1
     
    # sampled_subnets = sampled_image_subnets*12 + (12 - sampled_image_subnets)
    sampled_subnets.append(0)
    return sampled_subnets
                           
def get_sampling_function(args):
    if args.sampling_function == 'randomized_sampling':
        return get_random_subnets()
    elif args.sampling_function == 'big_small_sampling':
        return get_big_small_subnets()
    elif args.sampling_function == 'no_sampling':
        return get_no_sampling_subnets()

def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{CFG.captions_path}/captions.csv")
    # print(CFG.debug)
    max_id = dataframe.index.max() + 1 if not CFG.debug else 100
    # print(max_id)
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe.index.isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe.index.isin(valid_ids)].reset_index(drop=True)

    # print(len(train_dataframe))
    # print(len(valid_dataframe))
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
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

#comment out for the project
# def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
#     loss_meter = AvgMeter()
#     tqdm_object = tqdm(train_loader, total=len(train_loader))
#     for batch in tqdm_object:
#         batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
#         loss = model(batch)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if step == "batch":
#             lr_scheduler.step()

#         count = batch["image"].size(0)
#         loss_meter.update(loss.item(), count)

#         tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
#     return loss_meter

#Uncomment for weight shared training
def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        subnet_sampling_function = get_sampling_function(args)
        subnets = subnet_sampling_function

#         # if args.sam
#         # subnets = 
#         # print("Subnets:", subnets)
        total_loss = 0
        for subnet_no in subnets:
            # print(model)
            model.change_image_encoder_subnet(int(subnet_no/12))
            model.change_text_encoder_subnet(int(subnet_no%9))

            # print(model.eval())
            loss = model(batch)/len(subnets)
            total_loss += model(batch)/len(subnets)
            loss.backward()
        
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(total_loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter
# def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
#     loss_meter = AvgMeter()
#     tqdm_object = tqdm(train_loader, total=len(train_loader))
#     for batch in tqdm_object:
#         batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
#         optimizer.zero_grad()

#         subnet_sampling_function = get_sampling_function(args)
#         subnets = subnet_sampling_function

#         # if args.sam
#         # subnets = 
#         # print("Subnets:", subnets)
#         for subnet_no in subnets:
#             # print(model)
#             model.change_image_encoder_subnet(int(subnet_no/12))
#             model.change_text_encoder_subnet(int(subnet_no%12))

#             # print(model.eval())
#             loss = model(batch)/len(subnets)
#             loss.backward()
        
#         optimizer.step()
#         if step == "batch":
#             lr_scheduler.step()

#         count = batch["image"].size(0)
#         loss_meter.update(loss.item(), count)

#         tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
#     return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


# def sample_models():
#     # pairs = [(x, y) for x in range(0, 12) for y in range(0, 12)]
#     random_integers = random.sample(range(144), 4)
#     print("random_integers: ", random_integers)
#     return [CLIPModel(submodel=random_integers[0]).to(CFG.device), 
#     CLIPModel(submodel=random_integers[1]).to(CFG.device),
#     CLIPModel(submodel=random_integers[2]).to(CFG.device),
#     CLIPModel(submodel=random_integers[3]).to(CFG.device)]

def main():
    train_df, valid_df = make_train_valid_dfs()
    # tokenizer = BertTokenizer.from_pretrained(CFG.text_tokenizer)
    tokenizer = BertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")


    model = CLIPModel().to(CFG.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    print(CFG.epochs)
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), f'Best_ResnetDynabert_{args.sampling_function}.pt')
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)



if __name__ == "__main__":
    main()
