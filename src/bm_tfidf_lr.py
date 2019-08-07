""" Base Line """

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.utils.word2vec import multiclass_logloss
from sklearn import preprocessing, decomposition
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from sklearn.metrics import accuracy_score


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


stwlist = [line.strip() for line in open('../data/stop_words.txt', 'r', encoding='utf-8').readlines()]
tfv = NumberNormalizingVectorizer(min_df=3,
                                  max_df=0.5,
                                  max_features=None,
                                  ngram_range=(1, 2),
                                  use_idf=True,
                                  smooth_idf=True,
                                  stop_words=stwlist)

data = pd.read_csv('sentence_label2.txt', header=0)

lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data['label'].values)

xtrain, xvalid, ytrain, yvalid = train_test_split(data['sentence'].values, y,
                                                  stratify=y,
                                                  random_state=42,
                                                  test_size=0.1, shuffle=True)

tfv.fit(list(xtrain) + list(xvalid))
xtrain_tfv = tfv.transform(xtrain)
xvalid_tfv = tfv.transform(xvalid)

# 使用TF-IDF来fit训练集和测试集（半监督学习）
# 利用提取的TFIDF特征来fit一个简单的Logistic Regression
# clf = LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial')
# clf.fit(xtrain_tfv, ytrain)
# predictions = clf.predict_proba(xvalid_tfv)
# yvalid = np.array(yvalid)
#
# # logloss: 0.004
# print("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# # 使用SVD进行降维，components设为120，对于SVM来说，SVD的components的合适调整区间一般为120~200
# svd = decomposition.TruncatedSVD(n_components=120)
# svd.fit(xtrain_tfv)
# xtrain_svd = svd.transform(xtrain_tfv)
# xvalid_svd = svd.transform(xvalid_tfv)
#
# # 对从SVD获得的数据进行缩放
# scl = preprocessing.StandardScaler()
# scl.fit(xtrain_svd)
# xtrain_svd_scl = scl.transform(xtrain_svd)
# xvalid_svd_scl = scl.transform(xvalid_svd)
#
# # 调用下SVM模型
# clf = SVC(C=1.0, probability=True) # since we need probabilities
# clf.fit(xtrain_svd_scl, ytrain)
# predictions = clf.predict_proba(xvalid_svd_scl)
#
# # logloss: 0.002
# print("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# #利用提取的TFIDF特征来fitNaive Bayes
# clf = MultinomialNB()
# clf.fit(xtrain_tfv, ytrain)
# predictions = clf.predict_proba(xvalid_tfv)
#
# # logloss: 0.001
# print("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# 基于tf-idf特征，使用xgboost
clf = xgb.XGBClassifier(max_depth=4, n_estimators=200, colsample_bytree=0.8,
                        subsample=0.8, nthread=10, learning_rate=0.3)
clf.fit(xtrain_tfv.tocsc(), ytrain)
# predictions = clf.predict_proba(xvalid_tfv.tocsc())
#
# logloss: 0.000
# print("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

y_pred = clf.predict(xvalid_tfv)

# print(yvalid[:50])
# print(y_pred[:50])

re = pd.DataFrame(yvalid, columns=['yvalid'])
re['y_pred'] = y_pred

# re.to_csv('result.csv')

print("accuracy_score:", accuracy_score(yvalid, y_pred))

# svd = decomposition.TruncatedSVD(n_components=120)
# svd.fit(xtrain_tfv)
# xtrain_svd = svd.transform(xtrain_tfv)
# xvalid_svd = svd.transform(xvalid_tfv)
# # 基于tf-idf的svd特征，使用xgboost
# clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
#                         subsample=0.8, nthread=10, learning_rate=0.1)
# clf.fit(xtrain_svd, ytrain)
# predictions = clf.predict_proba(xvalid_svd)
#
# # logloss: 0.000
# print("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# svd = decomposition.TruncatedSVD(n_components=120)
# svd.fit(xtrain_tfv)
# xtrain_svd = svd.transform(xtrain_tfv)
# xvalid_svd = svd.transform(xvalid_tfv)
# scl = preprocessing.StandardScaler()
# scl.fit(xtrain_svd)
# xtrain_svd_scl = scl.transform(xtrain_svd)
# xvalid_svd_scl = scl.transform(xvalid_svd)
# # 再对经过数据标准化(Scaling)的tf-idf-svd特征使用xgboost
# clf = xgb.XGBClassifier(nthread=10)
# clf.fit(xtrain_svd_scl, ytrain)
# predictions = clf.predict_proba(xvalid_svd_scl)
#
# # logloss: 0.000
# print("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

