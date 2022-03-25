from utils.Dataset import HappyWhaleDataset
from torch.utils.data import DataLoader
from config import config

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)

def get_loader(df, split):
    df_train = df.loc[df['kfold'] != split]
    df_val = df.loc[df['kfold'] == split]
    df_val = df_val.sample(int(len(df_val)*0.1))

    train_dataset = HappyWhaleDataset(df_train, True)
    val_dataset = HappyWhaleDataset(df_val, True)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], pin_memory=True, shuffle=True)
    val_dataloadet = DataLoader(val_dataset, batch_size=config['batch_size'], pin_memory=True, shuffle=True)

    return train_dataloader, val_dataloadet