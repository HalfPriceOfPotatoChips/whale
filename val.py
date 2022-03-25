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

train_dataset = HappyWhaleDataset(train_csv, trainFlag=True)
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], pin_memory=True)

model = HappyWhaleModel(config['model_name'], config['num_class'], config['embedding_size'])
model.load_state_dict(torch.load(config['model_path']))
model = model.to(config['device'])

def inference(model, dataloader, device):
    model.eval()
    outputList=[]
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (img, label) in bar:
        img = img.to(device)
        label = label.to(device)
        _, outputs = model(img,label)
        outputList.extend(outputs.cpu().detach().numpy())
    return outputList


# train_embadding = np.array(inference(model, train_dataloader, config['device']))
# np.save('train_embadding.npy', train_embadding)
train_embadding = np.load('train_embadding.npy')
# df_train_embadding = pd.DataFrame(pd.Series(inference(model, train_dataloader, config['device'])), columns=['embadding'])
# df_train_embadding = pd.DataFrame(np.array(a), columns=['embadding'])

# df_train_embadding = pd.DataFrame(np.zeros((len(a), 1)), columns=['embadding'])
# df_train_embadding['embadding'] = np.array(a, dtype=list)

# train_csv = pd.concat([train_csv[0: 50], df_train_embadding], axis=1)
# train_csv.to_csv('archive/inference.csv')

def PredictGrid(train_csv, fold, thres):
    train_index = train_csv[train_csv['kfold'] != fold].index
    val_index = train_csv[train_csv['kfold'] == fold].index

    df_train = train_csv.iloc[train_index].reset_index(drop=True)
    df_val = train_csv.iloc[val_index].reset_index(drop=True)
    df_val_labels = df_val.individual_id.values

    # train_emb = df_train['embadding'].to_numpy()
    # val_emb = df_val['embadding'].to_numpy()
    train_emb = train_embadding[train_index]
    val_emb = train_embadding[val_index]

    knn_final_model = NearestNeighbors(n_neighbors=50)
    knn_final_model.fit(train_emb)

    D, I = knn_final_model.kneighbors(val_emb)

    print("Distances shape:", D.shape, "\n" +
            "Index shape:", I.shape)

    # C = 1 - D
    # preds = []
    #
    # for i, img_id in tqdm(enumerate(df_val['image'])):
    #     labels = df_train.loc[I[i], ['individual_id']].values.squeeze(axis=1)
    #     conf = C[i]
    #     subset_preds = pd.DataFrame(np.stack([labels, conf], axis=1),
    #                                 columns=['label', 'confidence'])
    #     # subset_preds.append({'label': 'new_individual', 'confidence': thres}, ignore_index=True)
    #     subset_preds['img_id'] = img_id
    #     preds.append(subset_preds)
    #
    # preds = pd.concat(preds).reset_index(drop=True)
    #
    # preds = preds.groupby(['img_id', 'label'])['confidence'].max().reset_index()
    #
    # predictions = {}
    #
    # for i, row in tqdm(preds.iterrows()):
    #     img_id = row['img_id']
    #     label = row['label']
    #     conf = row['confidence']
    #
    #     if img_id in predictions:
    #         if len(predictions[img_id]) != 5:
    #             predictions[img_id].insert(-1, label) if conf > thres else predictions[img_id].append(label)
    #         else:
    #             continue
    #     elif conf > thres:
    #         predictions[img_id] = [label, 'new_individual']
    #     else:
    #         predictions[img_id] = ['new_individual', label]

    preds = []

    for i, img_id in tqdm(enumerate(df_val['image'])):
        labels = df_train.loc[I[i], ['individual_id']].values.squeeze(axis=1)
        distances = D[i]
        subset_preds = pd.DataFrame(np.stack([labels, distances], axis=1),
                                    columns=['label', 'distances'])
        # subset_preds.append({'label': 'new_individual', 'confidence': thres}, ignore_index=True)
        subset_preds['img_id'] = img_id
        preds.append(subset_preds)

    preds = pd.concat(preds).reset_index(drop=True)

    preds = preds.groupby(['img_id', 'label'])['distances'].max().reset_index()

    predictions = {}

    for i, row in tqdm(preds.iterrows()):
        img_id = row['img_id']
        label = row['label']
        distance = row['distances']

        if img_id in predictions:
            # If total preds for this image_id are < 5 then add, else continue
            if len(predictions[img_id]) != 5:
                predictions[img_id].append(label)
            else:
                continue
            # If the distance is greater than thresh add prediction + "new_individual"
        elif distance > thres:
            predictions[img_id] = [label, "new_individual"]
        else:
            predictions[img_id] = ["new_individual", label]

    sample_list = ['37c7aba965a5', '114207cab555', 'a6e325d8e924', '19fbb960f07d', 'c995c043c353']

    for image_id, preds in tqdm(predictions.items()):
        if len(preds) < 5:
            remaining = [individ_id for individ_id in sample_list if individ_id not in preds]
            preds.extend(remaining)
            predictions[image_id] = preds[:5]

    return df_val_labels, predictions

iteration=0
best_score = 0
best_thres = 0
for thres in np.arange(0.4, 0.9, 0.01):

    print("iteration ", iteration, " of ", len(np.arange(0.4, 0.9, 0.1)))
    avg_map5 = RunningAverage()

    for fold in range(config['n_splits']):
        if fold != 1:
            continue

        df_val_labels, predictions = PredictGrid(train_csv, fold, thres)
        map5 = map_per_set(df_val_labels, list(predictions.values()))
        avg_map5.update(map5)

    score = avg_map5()
    print('     score: ', score)
    if score > best_score:
        best_score, best_thres = score, thres
    iteration += 1

print('~'*5, 'best_thres: ', best_thres, '~'*5)
print('~'*5, 'best_score: ', best_score, '~'*5)
