""""
参考: https://github.com/YutaroOgawa/pytorch_advanced/blob/master/7_nlp_sentiment_transformer/
"""
import random

import MeCab
import torchtext
from torchtext.vocab import Vectors


def get_DataLoaders(max_length=400, min_freq=5, batch_size=24):

    def wakati(text):
        tagger = MeCab.Tagger('/usr/local/lib/mecab/dic/mecab-ipadic-neologd/')
        tagger.parse('')
        node = tagger.parseToNode(text)
        word_list = []
        while node:
            pos = node.feature.split(",")[0]
            if pos in ['名詞', '動詞', '形容詞']:  # 対象とする品詞
                word = node.surface
                if len(word) > 0 and word not in ['\u3000', 'それ', 'てる', 'よう', 'こと', 'の', 'し', 'い', 'ん', 'さ', 'て', 'せ', 'れ']  and word != ',':
                    word_list.append(word)
            node = node.next
        return word_list

    TEXT = torchtext.data.Field(
        sequential=True,  # 可変長のtext
        tokenize=wakati,  # 前処理&tokenizeする関数
        use_vocab=True,
        lower=True,  # 小文字に変換
        include_lengths=True,  # 長さの保存.padding済みの行列とtextの長さを表す配列をreturn
        batch_first=True,  # バッチの次元を[0]
        fix_length=max_length,  # 最大長
        init_token="<cls>",  # 文頭を埋める
        eos_token="<eos>")
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
    print('building_torchtext.data.TabularDataset...')
    train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path='./',
        train='train.csv', test='test.csv',
        format='csv',
        fields=[('Text', TEXT), ('Label', LABEL)]
    )
    train_ds, val_ds = train_val_ds.split(split_ratio=0.8, random_state=random.seed(1234))
    print('train_val_len:', len(train_val_ds))
    print('train_len:', len(train_ds))
    print('val_len:', len(val_ds))
    print('test_len:', len(test_ds))

    TEXT.build_vocab(train_val_ds, min_freq=min_freq)

    print('reading_pretrained_vec_model')
    pretrained_vectors = Vectors(name='./vec/japanese_word2vec_vectors.vec')
    TEXT.build_vocab(train_ds, vectors=pretrained_vectors, min_freq=min_freq)
    print('shape_of_vocab:', TEXT.vocab.vectors.shape)

    train_dl = torchtext.data.Iterator(train_ds, batch_size=batch_size, train=True)

    val_dl = torchtext.data.Iterator(val_ds, batch_size=batch_size, train=False, sort=False)

    test_dl = torchtext.data.Iterator(test_ds, batch_size=batch_size, train=False, sort=False)

    return train_dl, val_dl, test_dl, TEXT

