""" Text-Classify by DeepLearning """

import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.utils.word2vec import sent2vec
from src.utils.matplot import draw_acc
import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
import gensim


def number_normalizer(tokens):
    """ 将所有数字标记映射为一个占位符（Placeholder）。
        对于许多实际应用场景来说，以数字开头的tokens不是很有用，但这样tokens的存在也有一定相关性。
        通过将所有数字都表示成同一个符号，可以达到降维的目的。
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super(NumberNormalizingVectorizer, self).build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))


def word_vec():
    stwlist = [line.strip() for line in open('../data/stop_words.txt', 'r', encoding='utf-8').readlines()]
    tfv = NumberNormalizingVectorizer(min_df=3,
                                      max_df=0.5,
                                      max_features=None,
                                      ngram_range=(1, 2),
                                      use_idf=True,
                                      smooth_idf=True,
                                      stop_words=stwlist)

    data = pd.read_csv('./tmp_files/sentence_label2.txt', header=0)
    # data = data.sample(frac=0.1)

    lbl_enc = preprocessing.LabelEncoder()
    y = lbl_enc.fit_transform(data['label'].values)

    xtrain, xvalid, ytrain, yvalid = train_test_split(data['sentence'].values, y,
                                                      stratify=y,
                                                      random_state=42,
                                                      test_size=0.1, shuffle=True)

    tfv.fit(list(xtrain) + list(xvalid))
    X = data["sentence"]
    X = [i.split() for i in X]

    model = gensim.models.Word2Vec(X, min_count=5, window=8, size=100)

    xtrain_w2v = [sent2vec(x, model) for x in tqdm(xtrain)]
    xvalid_w2v = [sent2vec(x, model) for x in tqdm(xvalid)]

    xtrain_w2v = np.array(xtrain_w2v)
    xvalid_w2v = np.array(xvalid_w2v)
    ytrain_np = np.array(ytrain)
    yvalid_np = np.array(yvalid)

    np.save('./tmp_files/xtrain_w2v.npy', xtrain_w2v)
    np.save('./tmp_files/xvalid_w2v.npy', xvalid_w2v)
    np.save('./tmp_files/ytrain_np.npy', ytrain_np)
    np.save('./tmp_files/yvalid_np.npy', yvalid_np)

    return xtrain_w2v, xvalid_w2v, ytrain, yvalid


def nn_model(x_train, x_valid, y_train, y_valid):
    # 在使用神经网络前，对数据进行缩放
    scl = preprocessing.StandardScaler()
    xtrain_w2v_scl = scl.fit_transform(x_train)
    xvalid_w2v_scl = scl.transform(x_valid)

    # 对标签进行binarize处理
    ytrain_enc = np_utils.to_categorical(y_train)
    yvalid_enc = np_utils.to_categorical(y_valid)

    # # 创建1个3层的序列神经网络（Sequential Neural Net）
    model = Sequential()

    # input_dim:指定输入尺寸   第一层是Dense层（全连接层）输入的是维度为1*100的列向量（input_dim=100）
    model.add(Dense(300, input_dim=100, activation='relu'))
    # Dropout层添加到模型的现有层和之前的输出层之间
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(4))
    model.add(Activation('softmax'))

    # 模型编译   损失函数选择的是交叉熵（loss=keras.losses.categorical_crossentropy）
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_train = model.fit(xtrain_w2v_scl, y=ytrain_enc, batch_size=64,
              epochs=20, verbose=1,
              validation_data=(xvalid_w2v_scl, yvalid_enc))

    return model_train


if __name__ == '__main__':
    # xtrain_w2v, xvalid_w2v, ytrain, yvalid = word_vec()
    xtrain_w2v = np.load('./tmp_files/xtrain_w2v.npy')
    xvalid_w2v = np.load('./tmp_files/xvalid_w2v.npy')
    ytrain = np.load('./tmp_files/ytrain_np.npy')
    yvalid = np.load('./tmp_files/yvalid_np.npy')

    model_train = nn_model(xtrain_w2v, xvalid_w2v, ytrain, yvalid)
    draw_acc(model_train)
