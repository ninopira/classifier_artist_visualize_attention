import csv
import os
import random

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


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
    ax.plot(df['epoch'], df['train_loss'], label='train_loss')
    ax.plot(df['epoch'], df['val_loss'], label='val_loss')
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    fig.savefig(png_path)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Liner層の初期化
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


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
    d_model=200,
    max_seq_len=400,
    output_dim=3)
print(net)


print('learning_start...')
# NNの初期化
net.train()
net.net3_1.apply(weights_init)
net.net3_2.apply(weights_init)

# GPU関連
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('devive：', device)
net.to(device)

# ネットワークがある程度固定であれば、高速化させる
torch.backends.cudnn.benchmark = True

# 学習ログ関連
train_loss_list = []
val_loss_list = []
result_dir = './result'
os.makedirs(result_dir, exist_ok=True)
csv_path = os.path.join(result_dir, 'history.csv')
png_path = os.path.join(result_dir, 'history.png')
with open(csv_path, 'w') as f:
    writer = csv.writer(f)
    header = ['epoch', 'train_loss', 'val_loss']
    writer.writerow(header)

# loss / optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-5)

# epochのループ
num_epochs = 2000
for epoch in range(num_epochs):
    # epochごとの訓練と検証のループ
    for phase in ['train', 'val']:
        if phase == 'train':
            net.train()
        else:
            net.eval()

        epoch_loss = 0.0  # epochの損失和
        epoch_corrects = 0  # epochの正解数

        # データローダーからミニバッチを取り出すループ
        for batch in tqdm((dataloaders_dict[phase])):
            # GPUが使えるならGPUにデータを送る
            inputs = batch.Text[0].to(device)
            labels = batch.Label.to(device)

            # optimizerを初期化
            optimizer.zero_grad()

            # 順伝搬（forward）計算
            with torch.set_grad_enabled(phase == 'train'):
                # mask作成
                input_pad = 1  # 単語のIDにおいて、'<pad>'
                input_mask = (inputs != input_pad)

                # Transformerに入力
                outputs, _, _ = net(inputs, input_mask)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                # 訓練時はバックプロパゲーション
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # 結果の計算
                epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                # 正解数の合計を更新
                epoch_corrects += torch.sum(preds == labels.data)

        # epochごとのlossと正解率
        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
        epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
        if phase == 'train':
            train_loss = epoch_loss
        else:
            val_loss = epoch_loss
        print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs, phase, epoch_loss, epoch_acc))
    # ep_end
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, train_loss, val_loss])
    if (epoch+1) % 5 == 0:
        plot_history(csv_path, png_path)
        model_name = 'ep_{}_weights.pth'.format(epoch+1)
        model_path = os.path.join(result_dir, model_name)
        torch.save(net.state_dict(), model_path)

print('learning_done')

