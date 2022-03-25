import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import time
import torch
from torch import nn
from utils.Dataset import HappyWhaleDataset
from utils.util import *
from utils.metric import map_per_set
from config import config
from torch.utils.data import DataLoader
from utils.model import HappyWhaleModel

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors


train_csv = pd.read_csv('archive/train_final.csv')
test_csv = pd.read_csv('archive/test_final.csv')

train_dataset = HappyWhaleDataset(test_csv, trainFlag=False)
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], pin_memory=True)

model = HappyWhaleModel(config['model_name'], config['num_class'], config['embedding_size'])
model.load_state_dict(torch.load(config['model_path']))
model = model.to(config['device'])

train_embedding = np.load('train_embadding.npy')

def inference(model, dataloader, device):
    model.eval()
    outputList=[]
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, img in bar:
        img = img.to(device)

        outputs = model(img)
        outputList.extend(outputs.cpu().detach().numpy())
    return outputList

test_embedding = np.array(inference(model, train_dataloader, config['device']))

knn_final_model = NearestNeighbors(n_neighbors=50)
knn_final_model.fit(train_embedding)

# D, I = knn_final_model.kneighbors(test_embedding)
# C = 1 - D
#
# test_images = test_csv["image"].tolist()
#
# test_df = []
# for k, image_id in tqdm(enumerate(test_images)):
#     # Get individual_id & distances for the observation
#     individual_id = train_csv.loc[I[k], ['individual_id']].values.squeeze(axis=1)
#     score = C[k]
#     # Create a df subset with this info
#     subset_preds = pd.DataFrame(np.stack([individual_id, score], axis=1),
#                                 columns=['individual_id','score'])
#     subset_preds['image_id'] = image_id
#     test_df.append(subset_preds)
#
# test_df = pd.concat(test_df).reset_index(drop=True)
# test_df = test_df.groupby(['image_id', 'individual_id'])['score'].max().reset_index()

# thres = 0.5
# predictions = {}
#
# for i, row in tqdm(test_df.iterrows()):
#     image_id = row['image_id']
#     individual_id = row['individual_id']
#     conf = row['score']
#
#     if image_id in predictions:
#         if len(predictions[image_id]) != 5:
#             predictions[image_id].insert(-1, individual_id) if conf > thres else predictions[image_id].append(individual_id)
#         else:
#             continue
#     elif conf > thres:
#         predictions[image_id] = [individual_id, 'new_individual']
#     else:
#         predictions[image_id] = ['new_individual', individual_id]
#
# sample_list = ['37c7aba965a5', '114207cab555', 'a6e325d8e924', '19fbb960f07d', 'c995c043c353']
#
# for image_id, preds in tqdm(predictions.items()):
#     if len(preds) < 5:
#         remaining = [individ_id for individ_id in sample_list if individ_id not in preds]
#         preds.extend(remaining)
#         predictions[image_id] = preds[:5]

D, I = knn_final_model.kneighbors(test_embedding)

test_images = test_csv["image"].tolist()

test_df = []
for k, image_id in tqdm(enumerate(test_images)):
    # Get individual_id & distances for the observation
    individual_id = train_csv.loc[I[k], ['individual_id']].values.squeeze(axis=1)
    distances = D[k]
    # Create a df subset with this info
    subset_preds = pd.DataFrame(np.stack([individual_id, distances], axis=1),
                                columns=['individual_id','distances'])
    subset_preds['image_id'] = image_id
    test_df.append(subset_preds)

test_df = pd.concat(test_df).reset_index(drop=True)
test_df = test_df.groupby(['image_id', 'individual_id'])['distances'].max().reset_index()

predictions = {}
thresh = 5

for k, row in tqdm(test_df.iterrows()):
    image_id = row["image_id"]
    individual_id = row["individual_id"]
    distance = row["distances"]

    # If the image_id has already been added in predictions before
    if image_id in predictions:
        # If total preds for this image_id are < 5 then add, else continue
        if len(predictions[image_id]) != 5:
            predictions[image_id].append(individual_id)
        else:
            continue
    # If the distance is greater than thresh add prediction + "new_individual"
    elif distance > thresh:
        predictions[image_id] = [individual_id, "new_individual"]
    else:
        predictions[image_id] = ["new_individual", individual_id]

# Fill in all lists that have less than 5 predictions as of yet
sample_list = ['37c7aba965a5', '114207cab555', 'a6e325d8e924', '19fbb960f07d', 'c995c043c353']

for image_id, preds in tqdm(predictions.items()):
    if len(preds) < 5:
        remaining = [individ_id for individ_id in sample_list if individ_id not in preds]
        preds.extend(remaining)
        predictions[image_id] = preds[:5]

predictions = pd.Series(predictions).reset_index()
predictions.columns = ['image','predictions']
predictions['predictions'] = predictions['predictions'].apply(lambda x: ' '.join(x))
predictions.to_csv('submission.csv',index=False)

predictions.head()