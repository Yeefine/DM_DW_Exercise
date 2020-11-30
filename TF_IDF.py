# -*- coding: utf-8 -*-
# @Time:  2020/11/17 20:58
# @Author : Yeefine
# @File : TF_IDF.py
# @Software : PyCharm

import math
from collections import  defaultdict
import operator
import os

"""
函数说明：特征选择TF-IDF算法
Parameters:
     list_words:词列表
Returns:
     dict_feature_select:特征选择词字典
"""


def feature_select(list_words):
    # 总词频统计
    doc_frequency = defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i] += 1
    # 计算每个词的TF值
    word_tf = {}  # 存储没个词的tf值
    num = sum(doc_frequency.values())
    for i in doc_frequency:
        word_tf[i] = doc_frequency[i] / float(num)
    # 计算每个词的IDF值
    doc_num = len(list_words)
    word_idf = {}  # 存储每个词的idf值
    word_doc = defaultdict(int)  # 存储包含该词的文档数
    cnt = 1

    for i in list_words:
        wordSet = set(i)
        for j in wordSet:
            word_doc[j] += 1

    for i in doc_frequency:
        word_idf[i] = math.log(doc_num / float(word_doc[i] + 1))

    # 计算每个词的TF*IDF的值
    word_tf_idf = {}
    for i in doc_frequency:
        word_tf_idf[i] = word_tf[i] * word_idf[i]
    # 对字典按值由大到小排序
    dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
    return dict_feature_select


if __name__ == '__main__':
    text = []
    dir = os.listdir("..\\CutDataSet")
    text=[]
    for filename in dir:
        with open("..\\CutDataSet\\"+filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                words = line.rstrip("\n").strip(" ").split(" ")
                while '' in words:
                    words.remove('')
                text.append(words)
    features = feature_select(text)
    cnt = 0
    bestFeatures = []
    features = dict(features)


    for key in features.keys():     # ['月', '日', '中国', '年', '中', '上', '新', '人', '都', '不', '10', '大', '后', '美国', '近日', '动画', '汽车', '会', '发布', '最']
        if cnt >= 40000: break
        bestFeatures.append(key)
        cnt += 1
    print(bestFeatures)
    # print(len(bestFeatures))

    with open("..\\TF_IDF.txt", "w", encoding='utf-8') as f:
        totalWords = ""
        for word in bestFeatures:
            totalWords += word + " "
        f.write(totalWords)

    print("finish!")