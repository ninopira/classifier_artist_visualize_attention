import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model import TransformerClassification
from dataloader import get_DataLoaders
from sklearn.metrics import confusion_matrix

label2artist = {
    0: 'B\'z',
    1: 'GLAY',
    2: 'Mr.children',
}


def print_cmx(y_true, y_pred, save_path):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cmx, annot=True, cmap='YlGnBu')
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.savefig(save_path)


print('building_dataloader...')
train_dl, val_dl, test_dl, TEXT = get_DataLoaders(max_length=300, min_freq=5, batch_size=24)
print('builiding_model...')

net = TransformerClassification(
    text_embedding_vectors=TEXT.vocab.vectors,
    d_model=200,
    max_seq_len=300,
    output_dim=3)
# print(net)

# PyTorchのネットワークパラメータのロード
load_path = './result/ep_550_weights.pth'
load_weights = torch.load(load_path)
net.load_state_dict(load_weights)

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.eval()   # モデルを検証モードに
net.to(device)

epoch_corrects = 0  # epochの正解数
scores = {}
all_labels = []
all_preds = []
for batch in (test_dl):  # testデータのDataLoader
    # GPUが使えるならGPUにデータを送る
    inputs = batch.Text[0].to(device)
    labels = batch.Label.to(device)

    # 順伝搬（forward）計算
    with torch.set_grad_enabled(False):
        # mask作成
        input_pad = 1
        input_mask = (inputs != input_pad)
        # Transformerに入力
        outputs, _, _ = net(inputs, input_mask)
        _, preds = torch.max(outputs, 1)  # ラベルを予測

        # 結果の計算
        # 正解数の合計を更新
        epoch_corrects += torch.sum(preds == labels.data)
        all_labels += [label2artist[label] for label in labels.data.cpu().numpy().tolist()]
        all_preds += [label2artist[pred] for pred in preds.cpu().numpy().tolist()]
        for label, pred in zip(labels.data, preds):
            label = int(label.cpu().numpy())
            pred = int(pred.cpu().numpy())
            label = label2artist[label]
            pred = label2artist[pred]
            try:
                scores['true:{}_pred:{}'.format(label, pred)] += 1
            except:
                scores['true:{}_pred:{}'.format(label, pred)] = 1

# 正解率
epoch_acc = epoch_corrects.double() / len(test_dl.dataset)
print(scores)
print_cmx(all_labels, all_preds, './result/coonfusion_matrix.png')

print('テストデータ{}個での正解率：{:.4f}'.format(len(test_dl.dataset), epoch_acc))


# HTMLを作成する関数を実装
def highlight(word, attn):
    "Attentionの値が大きいと文字の背景が濃い赤になるhtmlを出力させる関数"

    html_color = '#%02X%02X%02X' % (
        255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}"> {}</span>'.format(html_color, word)


def mk_html(index, batch, preds, normlized_weights_1, normlized_weights_2, TEXT):
    "HTMLデータを作成する"

    # indexの結果を抽出
    sentence = batch.Text[0][index]  # 文章
    label = batch.Label[index]  # ラベル
    pred = preds[index]  # 予測

    # indexのAttentionを抽出と規格化
    attens1 = normlized_weights_1[index, 0, :]  # 0番目の<cls>のAttention
    attens1 /= attens1.max()

    attens2 = normlized_weights_2[index, 0, :]  # 0番目の<cls>のAttention
    attens2 /= attens2.max()

    # ラベルと予測結果を文字に置き換え
    if label == 0:
        label_str = 'B\'z'
    elif label == 1:
        label_str = 'GLAY'
    else:
        label_str = 'Mr.Children'

    if pred == 0:
        pred_str = 'B\'z'
    elif pred == 1:
        pred_str = 'GLAY'
    else:
        pred_str = 'Mr.Children'

    # 表示用のHTMLを作成する
    html = '正解ラベル：{}<br>推論ラベル：{}<br><br>'.format(label_str, pred_str)

    # 1段目のAttention
    html += '[TransformerBlockの1段目のAttentionを可視化]<br>'
    for word, attn in zip(sentence, attens1):
        html += highlight(TEXT.vocab.itos[word], attn)
    html += "<br><br>"

    # 2段目のAttention
    html += '[TransformerBlockの2段目のAttentionを可視化]<br>'
    for word, attn in zip(sentence, attens2):
        html += highlight(TEXT.vocab.itos[word], attn)

    html += "<br><br>"

    return html

# ミニバッチの用意
batch = next(iter(test_dl))

inputs = batch.Text[0].to(device)
labels = batch.Label.to(device)
# mask作成
input_pad = 1  # 単語のIDにおいて、'<pad>': 1
input_mask = (inputs != input_pad)
# Transformerに入力
outputs, normlized_weights_1, normlized_weights_2 = net(inputs, input_mask)
_, preds = torch.max(outputs, 1)  # ラベルを予測
for i in range(10):
    html_output = mk_html(i, batch, preds, normlized_weights_1, normlized_weights_2, TEXT)  # HTML作成
    with open('./result/htmls/predict_{}.html'.format(i), 'wb') as file:
        file.write(html_output.encode('utf-8'))
