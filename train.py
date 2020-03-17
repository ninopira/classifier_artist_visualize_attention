import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext

from dataloader import get_DataLoaders

# set_seed
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# DataLoade
train_dl, val_dl, test_dl, TEXT = get_DataLoaders(max_length=400, min_freq=5, batch_size=24)
# 動作確認
batch = next(iter(train_dl))
print(batch.Text)
print(batch.Label)

import ipdb; ipdb.set_trace()

