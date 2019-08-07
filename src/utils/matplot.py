""" draw pics utils"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def draw_acc(model_train):
    # 绘图
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.plot(model_train.history['acc'], c='g', label='train')
    plt.plot(model_train.history['val_acc'], c='b', label='validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Model accuracy')

    plt.subplot(122)
    plt.plot(model_train.history['loss'], c='g', label='train')
    plt.plot(model_train.history['val_loss'], c='b', label='validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Model loss')

    plt.show()
