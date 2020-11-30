# -*- coding: utf-8 -*-
# @Time:  2020/11/14 21:56
# @Author : Yeefine
# @File : Confusion_Matrix.py
# @Software : PyCharm

import  matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn

# def drawPicture(TotalTypeList):
#     predVec = []
#     trueVec = []
#     for k in range(0, 10):
#         for i in TotalTypeList[k]:
#             predVec.append(i)
#             trueVec.append(k)
#     # cm = confusion_matrix(y_true=trueVec, y_pred=predVec, labels=['anima','car','ent','finalce','game','health','his','mil','sports','tech'])
#     cm = confusion_matrix(y_true=trueVec, y_pred=predVec)
#
#     plot_confusion_matrix(cm, ['anima','car','ent','finalce','game','health','his','mil','sports','tech'],'confusion matrix')


def drawPicture(predVec, trueVec):

    cm = confusion_matrix(y_true=trueVec, y_pred=predVec)
    plot_confusion_matrix(cm, ['anima','car','ent','finance','game','health','his','mil','sports','tech'],'confusion matrix')
    return cm


def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()