import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm
import time
import torch
from torch import nn
from utils.Dataset import HappyWhaleDataset
from utils.util import *
from utils.metric import map_per_set
from config import config
from torch.utils.data import DataLoader
from utils.model import HappyWhaleModel
from torch.optim import Adam, lr_scheduler, SGD

from torch.utils.tensorboard import SummaryWriter

train_csv = pd.read_csv('archive/train_final.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


def get_model_optimizer_criterion(model_path=None):

    model = HappyWhaleModel(config['model_name'], config['num_class'], config['embedding_size'])
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    #optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], momentum=0.5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['T_max'], eta_min=config['min_lr'])
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, scheduler, criterion

if not os.path.exists('runs/B6'):
    os.mkdir('runs/B6')
writer = SummaryWriter('runs/B6')

train_loss_avg = RunningAverage()
# train_map5_avg = RunningAverage()
val_loss_avg = RunningAverage()
# val_map5_avg = RunningAverage()

best_loss = 15
# best_map5 = 0.3

model, optimizer, scheduler, criterion = get_model_optimizer_criterion(config['model_path'])

for fold in range(config['n_splits']):

    train_dataloader, val_dataloadet = get_loader(train_csv, fold)
    print('-' * 8, f'Fold {fold}', '-' * 8)
    for_epochs = fold*config['epochs']
    train_step = for_epochs*len(train_dataloader)
    val_step = for_epochs*len(val_dataloadet)

    for epoch in range(config['epochs']):

        print("~" * 8, f"Epoch {epoch}", "~" * 8)
        model.train()
        now_train_step = epoch*len(train_dataloader) + train_step
        now_val_step = epoch*len(val_dataloadet) + val_step

        train_loss_avg = RunningAverage()
        val_loss_avg = RunningAverage()

        for i, (img, cls) in enumerate(train_dataloader):

            start = time.time()
            img = img.to(device)
            cls = cls.to(device)

            optimizer.zero_grad()
            pred, _ = model(img, cls)
            Before0 = list(model.parameters())[0].clone()
            Before2 = list(model.parameters())[-2].clone()
            Before1 = list(model.parameters())[-1].clone()

            loss = criterion(pred, cls)

            # map5 = map_per_set(pred, cls)
            # train_map5_avg.update(map5)

            loss.backward()

            # for name, parms in model.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
            #           ' -->grad_value:', parms.grad)

            optimizer.step()
            After0 = list(model.parameters())[0].clone()  # 获取更新后模型的第0层权重
            After2 = list(model.parameters())[-2].clone()
            After1 = list(model.parameters())[-1].clone()

            train_loss_avg.update(loss.item())
            writer.add_scalar('Loss_train/step', loss.item(), i + now_train_step)

            print('     ---- Train batch {} / {} : batch_loss = {:.7f}'.format(i, len(train_dataloader),
                                                                                               loss.item()))

            print('模型的第0层更新幅度：', torch.sum(After0 - Before0))
            print('模型的第-1层更新幅度：', torch.sum(After1 - Before1))
            print('模型的第-2层更新幅度：', torch.sum(After2 - Before2))
            delta_time = time.time() - start
            print('          batches processed in {:.2f} seconds'.format(delta_time))

        model.eval()
        with torch.no_grad():
            for i, (img, cls) in enumerate(val_dataloadet):
                start = time.time()
                img = img.to(device)
                cls = cls.to(device)

                pred, _ = model(img, cls)

                loss = criterion(pred, cls)
                val_loss_avg.update(loss.item())
                # map5 = map_per_set(pred, cls)
                # val_map5_avg.update(map5)

                writer.add_scalar('Loss_val/step', loss.item(), i + now_val_step)
                # writer.add_scalar('map5_val/step', map5, i)

                print('     ---- Val batch {} / {} : batch_loss = {:.7f}'.format(i, len(val_dataloadet),
                                                                                                 loss.item()))

                delta_time = time.time() - start
                print('          batches processed in {:.2f} seconds'.format(delta_time))

        scheduler.step()
        train_epoch_loss, val_epoch_loss, = train_loss_avg(), val_loss_avg()
        writer.add_scalar('Loss_train/epochs', train_epoch_loss, epoch + for_epochs)
        # writer.add_scalar('map5_train/epochs', train_epoch_map5, epoch)

        writer.add_scalar('Loss_val/epochs', val_epoch_loss, epoch + for_epochs)
        # writer.add_scalar('map5_val/epochs', val_epoch_map5, epoch)

        is_best_loss = val_epoch_loss < best_loss
        # is_best_map5 = val_epoch_map5 < best_map5
        if(is_best_loss):
            print("! Saving model in fold {} | epoch {} ...".format(fold, epoch), "\n")
            best_loss = val_epoch_loss
            if not os.path.exists('runs/weight'):
                os.mkdir('runs/weight')
            torch.save(model.state_dict(), f"runs/weight/EffNetB6_fold_{fold}_loss_{round(best_loss, 3)}.pt")
