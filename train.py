import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext

from dataloader import get_DataLoaders
from model import TransformerClassification

# set_seed
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


def plot_history(csv_path, png_path):
    df = pd.read_csv(csv_path)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df['epoch'], df['train_loss'], label='g_train')
    ax.plot(df['epoch'], df['val_loss'], label='d_train')
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    fig.savefig(png_path)


# DataLoader
print('building_dataloader...')
train_dl, val_dl, test_dl, TEXT = get_DataLoaders(max_length=400, min_freq=5, batch_size=24)
# 動作確認
# batch = next(iter(train_dl))
# print(batch.Text)
# print(batch.Label)
dataloaders_dict = {"train": train_dl, "val": val_dl}

print('builiding_model...')

net = TransformerClassification(
    text_embedding_vectors=TEXT.vocab.vectors, 
    d_model=300,
    max_seq_len=256,
    output_dim=3)
print(net)





