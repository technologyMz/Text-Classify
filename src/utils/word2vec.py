import gensim
import jieba
import pandas as pd
from gensim.models import word2vec
import numpy as np
import re
from string import digits

segmented_file = '../data/trainData.txt'
modelPath = 'word2vec.model'


def remove_urls(vTEXT):
    """
    删除字符串中的网址
    :param vTEXT:
    :return:
    """
    return re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)


def remove_digit(sentence):
    """
    删除字符串中的数字
    :param sentence:
    :return:
    """
    return sentence.translate(str.maketrans('', '', digits))


def stopwords_list(file_path):
    """
    加载停用词列表
    :param file_path:
    :return:
    """
    stopwords = [line.strip() for line in open(file_path, 'r', encoding='utf-8').readlines()]
    return stopwords


def move_stop_words(sentence):
    """
    去除停用词 for sentence
    :param sentence:
    :return:
    """
    stopwords = stopwords_list('../data/stop_words.txt')
    santi_words = [x for x in sentence if len(x) > 1 and x not in stopwords]
    return santi_words


def load_message_and_save(data, taget_file):
    """
    分词、去除停用词 并将结果写入文件
    :return:
    """
    # data = pd.read_csv("../data/xsms_order_db_t_platform_order2.csv")

    jieba.enable_parallel(4)

    with open(taget_file, 'w') as f2:
        for i in range(len(data["content"])):
            li = []
            sentence = data["content"][i]
            sentence = remove_urls(sentence)
            sentence = remove_digit(sentence)

            seg_list = jieba.cut(sentence)
            for w in seg_list:
                li.append(w)
            train_words = move_stop_words(li)
            line = ' '.join(train_words)
            f2.write(line+'\n')
    f2.close()


def load_message_and_save2(data, cols):
    """
    分词 & 去除停用词、网址、数字
    :return:
    """
    # data = pd.read_csv("../data/xsms_order_db_t_platform_order2.csv")

    jieba.enable_parallel(4)

    lines = []
    for i in range(len(data["content"])):
        li = []
        sentence = data["content"][i]
        sentence = remove_urls(sentence)
        sentence = remove_digit(sentence)

        seg_list = jieba.cut(sentence)
        for w in seg_list:
            li.append(w)
        train_words = move_stop_words(li)
        line = ' '.join(train_words)
        lines.append(line)
    return pd.DataFrame(lines, columns=cols)


def save_data(data):
    """
    保存list数据到文件
    :param data:
    :return:
    """
    words = ' '.join([' '.join(x) for x in data])
    segmented = open(segmented_file, "w", encoding="utf-8")
    segmented.write(words)
    segmented.flush()


def read_from_labeled_txt(filepath_dict):
    """
    加载打好标签的文件数据
    :param filepath_dict:
    :return:
    """
    df_list = []
    for source, file_path in filepath_dict.items():
        df_tmp = pd.read_csv(file_path, header=0)
        # df_tmp['label'] = [source] * len(df_tmp)
        df_list.append(df_tmp)
    re_df = df_list[0]
    for index in range(1, len(df_list)):
        re_df = re_df.append(df_list[index], ignore_index=True)
    return re_df
    # ['fuiorderid', 'fuimoblie', 'content', 'status', 'fstrsignname', 'fstrsendtime']


def train():
    """
    通过 word2vec 找出相似度最大的top10单词
    :return:
    """
    sentences = word2vec.LineSentence(segmented_file)
    model = gensim.models.Word2Vec(sentences, size=200, sg=1, iter=8)

    for key in model.similar_by_word('验证码', topn=10):
        print(key)


def multiclass_logloss(actual, predicted, eps = 1e-15):
    """对数损失度量（Logarithmic Loss  Metric）的多分类版本。
    :param actual: 包含actual target classes的数组
    :param predicted: 分类预测结果矩阵, 每个类别都有一个概率
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


def sent2vec(s, model):
    """
    将语句转化为一个标准化的向量（Normalized Vector）
    :param s: sentence
    :param model: gensim.models.Word2Vec
    :return:
    """
    # jieba.enable_parallel()
    words = str(s)
    # words = jieba.lcut(words)
    # words = move_stop_words(words)

    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())

