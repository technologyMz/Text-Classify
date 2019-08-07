#coding=utf-8
import jieba
import pandas as pd
import numpy as np
import logging
import time
from sklearn.svm import SVC
from sklearn import preprocessing, decomposition, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt


def logloss(actual, predicted, eps=1e-15):  # 评分函数 分数越低越好
    """对数损失度量（Logarithmic Loss  Metric）的多分类版本。
    :param actual: 包含actual target classes的数组
    :param predicted: 分类预测结果矩阵, 每个类别都有一个概率
    """
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2
    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


def fenci(data):
    logging.info("并行分词开始")
    jieba.enable_parallel() #并行分词 尽量在linux下使用
    data['content'] = data['content'].apply(lambda i:jieba.cut(i))
    data['content'] =[' '.join(i) for i in data['content']]

    writer = pd.ExcelWriter("fenciceshi"+'.xlsx')
    data.to_excel(writer,'page_1')
    writer.save()
    print("存储完成")


def number_normalizer(tokens):
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super(NumberNormalizingVectorizer, self).build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))


def FeatureProcessing(x_train):
    stwlist = [line.strip() for line in open('../data_test/zhongwentingyongci.txt', 'r', encoding='utf-8').readlines()]  # 载入停用词
    tfv = NumberNormalizingVectorizer(min_df=3,
                                      max_df=0.5,
                                      max_features=None,
                                      ngram_range=(1, 2),
                                      use_idf=True,
                                      smooth_idf=True,
                                      stop_words=stwlist)
    TFIDF_MODEL=tfv.fit(list(x_train))
    print ("TFIDF_MODEL的类型：",type(TFIDF_MODEL))
    joblib.dump(TFIDF_MODEL, "../model_test/" + 'TFIDF_MODEL.model')
    xtrain_tfv = tfv.transform(x_train)
    svd = decomposition.TruncatedSVD(n_components=120)
    TruncatedSVD_model=svd.fit(xtrain_tfv)
    joblib.dump(TruncatedSVD_model, "../model_test/" + 'TruncatedSVD_model.model')
    xtrain_svd = svd.transform(xtrain_tfv)
    scl = preprocessing.StandardScaler()
    StandardScaler_MODEL=scl.fit(xtrain_svd)
    joblib.dump(StandardScaler_MODEL, "../model_test/" + 'StandardScaler_MODEL.model')
    xtrain_svd_scl = scl.transform(xtrain_svd)
    return xtrain_svd_scl


def svm(x_train, y_train,param):
    param_c=param["svm__C"]
    param_kernel=param["svm__kernel"]
    clf = SVC(C=param_c, probability=True,kernel=param_kernel)  # since we need probabilities
    clf.fit(x_train, y_train)
    return clf


def dataPro():
    data_pass = pd.read_csv('../data_test/借款审批通过.csv',encoding='utf-8',engine='python')
    data_unpass = pd.read_csv('../data_test/借款拒绝进件.csv',encoding='utf-8',engine='python')
    data_Repayment=pd.read_csv('../data_test/还款成功.csv',encoding='utf-8',engine='python')
    data_Overdue=pd.read_csv('../data_test/逾期.csv',encoding='utf-8',engine='python')
    data=data_pass.append(data_unpass).append(data_Repayment).append(data_Overdue)[["status","content"]]

    jieba.enable_parallel()
    data['content'] = data['content'].apply(lambda i:jieba.cut(i))
    data['content'] =[' '.join(i) for i in data['content']]
    writer = pd.ExcelWriter("fenciceshi"+'.xlsx')
    data.to_excel(writer,'page_1')
    writer.save()
    print("分词存储完成")


def paramSelect(x_train,y_train,params):
    reParam={}
    # SVC:是一种基于libsvm的支持向量机，由于其时间复杂度为O(n^2)，所以当样本数量超过两万时难以实现。
    # probability = True：该参数表示是否启用概率估计。 这必须在调用fit()之前启用，并且会使fit()方法速度变慢。
    svm_model =SVC(probability=True)
    scorer = metrics.make_scorer(logloss, greater_is_better=False, needs_proba=True)
    clf = pipeline.Pipeline([('svm', svm_model)])
    # estimator：选择使用的分类器
    # param_grid：需要最优化的参数的取值，值为字典或者列表
    # scoring： 模型评价标准，默认None, 这时需要使用score函数；或者如scoring = 'roc_auc'，根据所选模型不同，评价准则不同
    # n_jobs：并行数
    # iid: 默认True, 为True时，默认为各个样本fold概率分布一致，误差估计为所有样本之和，而非各个fold的平均。

    # refit: 默认为True, 程序将会以交叉验证训练集得到的最佳参数，重新对所有可用的训练集与开发集进行，
    # 作为最终用于性能评估的最佳模型参数。即在搜索参数结束后，用最佳参数结果再次fit一遍全部数据集。

    # cv: 交叉验证参数，默认None，使用三折交叉验证。指定fold数量，默认为3，也可以是yield训练 / 测试数据的生成器。
    model = GridSearchCV(estimator=clf, param_grid=params, scoring=scorer,
                         verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)
    model.fit(x_train, y_train)
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        reParam[param_name] = best_parameters[param_name]
    return reParam


def getModel(params):
    modelPath = "../model_test/"
    data = pd.read_excel('fenciceshi.xlsx')  # 读取分词后的结果
    lbl_enc = preprocessing.LabelEncoder()
    LabelEncoder_model=lbl_enc.fit(data["status"].values)
    joblib.dump(LabelEncoder_model, "../model_test/" + 'LabelEncoder_model.model')
    y_train = lbl_enc.transform(data["status"].values)
    x_train=data["content"].values
    x_train = FeatureProcessing(x_train=x_train)
    acc_line(X_train=x_train,y_train=y_train) #绘制学习曲线

    reParam_time_start = time.time()
    reParam=paramSelect(x_train,y_train,params)
    reParam_time_end = time.time()
    print('超参数选择训练时间：', (reParam_time_end - reParam_time_start)/60, 'min')

    f = open('../data_test/paramter.txt', 'w')
    print("最优参数：",reParam)
    #训练svm模型并保存
    svm_model=svm(x_train=x_train, y_train=y_train, param=reParam)
    joblib.dump(svm_model,modelPath+'svm_model.model')
    logging.info("=====模型保存成功=====")


def acc_line(X_train,y_train):
    print ("y_train:",y_train)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20, random_state=1)
    pipe_svm = Pipeline([('svm',SVC(C=10, probability=True))])
    # 学习曲线
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_svm, X=X_train, y=y_train,train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
    # 统计结果
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    # 绘制效果
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    plt.show()
    plt.savefig("acc.png")


if __name__ == '__main__':
    #dataPro()#数据处理输入四种分类的数据文件 输出分词后的excel文件

    # 表示错误项的惩罚系数C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低；
    # 相反，减小C的话，容许训练样本中有一些误分类错误样本，泛化能力强。对于训练样本带有噪声的情况，一般采用后者，
    # 把训练样本集中错误分类的样本作为噪声。
    param = {'svm__C': [1.0, 10, 20, 30],               #设置调优的超参
    # kernel: 核函数
             'svm__kernel':['linear', 'poly', 'rbf']}
    getModel(params=param)#训练模型 输入分词后的excel文件，输出各阶段model 注：模型会自动使用最优参数

