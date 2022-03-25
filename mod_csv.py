import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

train = pd.read_csv("archive/train_final.csv")
test = pd.read_csv("archive/test2.csv")

df_train = train.loc[train['kfold']==0]
df_val = train.loc[train['kfold']!=0]


# train['path'] = 'archive/cropped_train_images/cropped_train_images/' + train['image']
# test['path'] = 'archive/cropped_test_images/cropped_test_images/' + test['image']
#
# for i, id in tqdm(enumerate(train['individual_id'].unique())):
#     train.loc[train['individual_id']==id, 'individual_key'] = i
# train['individual_key'] = train['individual_key'].astype(int)
#
# skf = StratifiedKFold(n_splits=5)
# for fold, (train_index, val_index) in tqdm(enumerate(skf.split(X=train, y=train['individual_key']))):
#     train.loc[val_index, 'kfold'] = fold
# train['kfold'] = train['kfold'].astype(int)
#
# train.to_csv('archive/train_final.csv')
# test.to_csv('archive/test_final.csv')