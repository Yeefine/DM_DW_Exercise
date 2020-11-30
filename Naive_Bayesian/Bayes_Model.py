# -*- coding: utf-8 -*-
# @Time:  2020/11/10 22:04
# @Author : Yeefine
# @File : Bayes_Model.py
# @Software : PyCharm

import os
import jieba
# from numpy import *
from Naive_Bayesian import Confusion_Matrix
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from collections import defaultdict
import math
import operator
from datetime import datetime


# 获取目录下的所有文件路径（二级目录）
def getFilePath(rootPath):
    filePathList = []
    files = os.listdir(rootPath)    # 返回一个列表，其中包含在目录条目的名称
    for f in files:
        filepath = rootPath + "\\" + f     # 返回一个列表，其中包含在目录条目的名称
        filePathList.append(filepath)
    return filePathList


# 获取停用词表的list
def getStopWords(file_stop):
    stopWords = []
    with open(file_stop, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stopWords.append(line.rstrip("\n"))
    return stopWords


# 使用jieba将句子分词，并根据停用词表删除无意义词，返回分词清洗后的list集合
def wordSegment(dirPath, stopWords):
    data = []
    cutwordslist = []
    with open(dirPath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data.append(line)

    for line in data:
        words = []
        list = jieba.cut(line)
        for word in list:
            if word not in stopWords:
                words.append(word.rstrip('\n'))
                # print(word)
        cutwordslist.append(words)
    return cutwordslist


# 获取单类别分割后的列表，  [["今天", "天气", "好"], [...], [...], ...]
def perTypeData(rootPath, file_stop):
    stopWords = getStopWords(file_stop)     # 获取停用词表的list
    # print(stopWords)
    filePathList = getFilePath(rootPath)        # 获取文件路径名的列表

    perTypeCutWordsList = []        # 单个类别的所有拆分后的句子

    for path in filePathList:
        print(path)
        cutwordslist = wordSegment(path, stopWords)
        perTypeCutWordsList.extend(cutwordslist)
    return perTypeCutWordsList


# 把每个文档对应的句子处理分割后写入新文档
def writeBackData(perTypeCutWordsList, filePath):
    with open(filePath, "w", encoding="utf-8") as f:
        for words in perTypeCutWordsList:
            str = ""
            for word in words:
                str += word + " "
            str += "\n"
            f.write(str)



# 获得分词后的需要写入的各类文件名
def newCutFilePath():
    rootpath = "..\\dataset"
    newRootpath = "..\\CutDataSet"
    filePathList = []
    files = os.listdir(rootpath)
    for f in files:
        filepath = newRootpath + "\\" + f  + ".txt"    # 返回一个列表，其中包含在目录条目的名称
        filePathList.append(filepath)
    return filePathList


# 将所有处理过的语句写入CutDataSet文件夹下的txt
def newCutFile():
    file_stop = "..\\stop_words_ch.txt"
    dirPath = getFilePath("..\\dataset")
    idx = 0
    fileTypeList = newCutFilePath()
    for dir in dirPath:
        perTypeCutWordsList = perTypeData(dir, file_stop)
        writeBackData(perTypeCutWordsList, fileTypeList[idx])
        idx += 1


# 按类别生成不重复词语列表，并写入VocabList下的txt
def createVocabList():
    files = os.listdir("..\\dataset")
    for name in files:
        filePath = "..\\CutDataSet\\" + name + ".txt"
        vocabSet = set([])
        with open(filePath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                word = line.rstrip("\n").split(" ")
                vocabSet = vocabSet | set(word)
        vocablist = list(vocabSet)
        str = ""
        with open("..\\VocabList\\" + name + ".txt", "w", encoding="utf-8") as f:
            for words in vocablist:
                str += words + " "
            f.write(str)


# 创建包含所有类别不同词语的词袋模型，并写入totalVocabList.txt
def createTotalVocabList():
    files = os.listdir("..\\VocabList")
    totalVocabSet = set([])
    for name in files:
        with open("..\\VocabList\\" + name, "r", encoding="utf-8") as f:
            lines = f.readlines()
            words = lines[0].split(" ")
            totalVocabSet = totalVocabSet | set(words)
    totalVocabList = list(totalVocabSet)
    str = ""
    with open("..\\totalVocabList.txt", "w", encoding="utf-8") as f:
        for word in totalVocabList:
            str += word + " "
        f.write(str)


"""
函数说明：特征选择TF-IDF算法
Parameters:
     list_words:词列表
Returns:
     bestFeatures：选取前40000个词作为最佳特征
"""
def feature_select():
    list_words = []
    dir = os.listdir("..\\CutDataSet")
    for filename in dir:
        with open("..\\CutDataSet\\"+filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                words = line.rstrip("\n").strip(" ").split(" ")
                while '' in words:
                    words.remove('')
                list_words.append(words)
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
    features = dict(dict_feature_select)
    bestFeatures = []
    for key in list(features.keys())[:40000]:   # 取前40000个词
        # if cnt >= 40000: break
        bestFeatures.append(key)
        # cnt += 1
    return bestFeatures             # ['月', '日', '中国', '年', '中', '上', '新', '人', '都', '不', '10', '大', '后', '美国', '近日', '动画', '汽车', '会', '发布', '最']


# 训练数据，使用每类数据的前50%训练，
# 返回[每类数据量的占比，每类中各词语的数量（出现在TF-IDF筛选出的词），每类的测试集数据，每类数据的总词数（出现在TF-IDF筛选出的词）+词袋的词数（TF-IDF筛选的总词数）]
def trainNB(bestFeatures):
    dirpath = "..\\CutDataSet"
    filenames = os.listdir(dirpath)
    totalNum = 0    # 所有句子的总个数
    for name in filenames:
        path = dirpath + "\\" + name
        with open(path, "r", encoding="utf-8") as f:
            words = f.readlines()
            totalNum += len(words)

    # # 读TF-IDF后选出的词语
    # with open("..\\TF_IDF.txt", "r", encoding='utf-8') as f:
    #     bestFeatures = f.read().strip(" ").split(" ")



    with open("..\\totalVocabList.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        totalVocabList = lines[0].strip().split(" ")[:-1]       # 词袋的词语总数

    EachTypePerc = []   # 每类数据量的占比（分母是总数据量）
    EachTypeWords = []  # 每类中各词语（出现在TF-IDF筛选后的词语中）的数量
    EachTypeTestData= []    # 每类的测试集数据
    EachTypeDataNum = []    # 每类数据的总词数+词袋词数


    for name in filenames:
        filepath = "..\\CutDataSet\\" + name
        file = []   # 存所有句子，每个句子被分词后存在一个list中
        with open(filepath, "r", encoding="utf-8") as f:
            words = f.readlines()
            for word in words:
                file.append(word.rstrip("\n").split(" ")[:-2])
        perTypeNum = len(file)
        p0 = perTypeNum / float(totalNum)   # 第一类的句子数占所有句子的比例
        EachTypePerc.append(p0)

        trainData = file[:int(perTypeNum*0.5)]      # 训练集， 取前50%的数据
        testData = file[int(perTypeNum*0.5):]       # 测试集， 取后50%的数据
        EachTypeTestData.append(testData)

        wordDict = {}       # 统计此种类中每个词的出现次数
        WordsNum = 0    # 统计此种类中所有词的总数 + 词袋的词数

        bestFeatures_map={}
        for best in bestFeatures:
            bestFeatures_map[best]=1

        for words in trainData:
            for word in words:
                if word not in bestFeatures_map.keys():
                    bestFeatures_map[word] = 0
        for words in trainData:
            for word in words:
                if bestFeatures_map[word] == 1 :
                    WordsNum += 1
                    num = wordDict.get(word, 0)
                    wordDict[word] = num+1

        EachTypeWords.append(wordDict)

        WordsNum += len(bestFeatures)
        EachTypeDataNum.append(WordsNum)

    return EachTypePerc, EachTypeWords, EachTypeTestData, EachTypeDataNum


# 输入一个句子的字典，判断各类别的概率，返回最可能的类别
def classifyNB(DictWords, EachTypeWords, EachTypeDataNum, pClass):
    size = len(EachTypeWords)
    p = 0.0
    type = 0
    for i in range(0, size):
        p0 = 0.0
        for word in DictWords:
            num = EachTypeWords[i].get(word, 0)
            p0 += np.log((num+1) / float(EachTypeDataNum[i]))   # add-one smoothing
        p0 += np.log(pClass[i])
        if i == 0:
            p = p0
            type = 0
        elif p0 > p:
            type = i
            p = p0
    return type


# 输入[每类数据量的占比，每类中各词语出现的次数，每类的测试集，每类的总词数+词袋的词数]
def testingNB(EachTypePerc, EachTypeWords, EachTypeTestData, EachTypeDataNum, bestFeatures):
    size = len(EachTypeTestData)
    TotalTypeList = []
    # with open("..\\TF_IDF.txt", "r", encoding='utf-8') as f:
    #     bestFeatures = f.read().strip(" ").split(" ")

    bestFeatures_map = {}
    for best in bestFeatures:
        bestFeatures_map[best] = 1

    for i in range(0, size):
        for words in EachTypeTestData[i]:
            for word in words:
                if word not in bestFeatures_map.keys():
                    bestFeatures_map[word] = 0

    for i in range(0, size):
        PredTypeList = []
        for words in EachTypeTestData[i]:
            DictWords = {}

            for word in words:
                if bestFeatures_map[word] == 1 :
                    num = DictWords.get(word, 0)
                    DictWords[word] = num+1
            type = classifyNB(DictWords, EachTypeWords, EachTypePerc, EachTypeDataNum)
            PredTypeList.append(type)
        TotalTypeList.append(PredTypeList)
    return TotalTypeList


# 混淆矩阵打印输出（横轴是预测值，纵轴是实际值）
def showConfusionMatrix(cm):
    print("混淆矩阵为：")
    labels = ['anima', 'car', 'ent', 'finance', 'game', 'health', 'his', 'mil', 'sports', 'tech']
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)
    print(pd.DataFrame(cm, labels, labels))


# 求各类的精确率、召回率、f1_score
def prec_recall_f1score(cm):
    data=[]
    col_sum = np.sum(cm, axis=0)  # 按列求和，用于求精确率
    row_sum = np.sum(cm, axis=1)  # 按行求和，用于求召回率
    for i in range(len(cm)):
        precision = cm[i][i] / float(col_sum[i])
        recall = cm[i][i] / float(row_sum[i])
        f1_score = (2 * precision * recall) / float(precision + recall)
        data.append([precision, recall, f1_score])
    labels = ['anima', 'car', 'ent', 'finance', 'game', 'health', 'his', 'mil', 'sports', 'tech']
    tags = ['precision', 'recall', 'f1-score']
    print(pd.DataFrame(data, labels, tags))

if __name__ == '__main__':

    # # 将所有处理过的语句写入CutDataSet文件夹下的txt （经过分词，去停用词）
    # newCutFile()

    # TF-IDF算法，选取前40000个词
    bestFeatures = feature_select()


    print("开始训练！")
    a = datetime.now()
    # 训练数据
    EachTypePerc, EachTypeWords, EachTypeTestData, EachTypeDataNum = trainNB(bestFeatures)
    b = datetime.now()
    print("训练结束！")
    print("训练耗时：", (b-a).seconds, "s")  # 9s

    # 测试数据，返回测试结果
    print("开始测试！")
    c = datetime.now()
    TotalTypeList = testingNB(EachTypePerc, EachTypeWords, EachTypeTestData, EachTypeDataNum, bestFeatures)
    d = datetime.now()
    print("测试结束！")
    print("测试耗时：", (d-c).seconds, "s")  # 104s


    # cnt = 0
    predVec = []    # 预测值列表
    trueVec = []    # 真实值列表
    for k in range(0, 10):
        for i in TotalTypeList[k]:
            predVec.append(i)
            trueVec.append(k)

    # # 将预测向量和真实值向量写入res.txt
    # with open("res.txt", "w", encoding="utf-8") as f:
    #     for i in range(len(predVec)):
    #         line = str(predVec[i]) + " " + str(trueVec[i]) + "\n"
    #         f.write(line)
    # print("write finish!")

    # 读取预测值与真实值
    # with open("res.txt", "r", encoding="utf-8") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         nums = line.rstrip("\n").split(" ")
    #         predVec.append(int(nums[0]))
    #         trueVec.append(int(nums[1]))

    cm = Confusion_Matrix.drawPicture(predVec, trueVec)
    print("朴素贝叶斯：")
    acc = accuracy_score(trueVec, predVec, normalize=True, sample_weight=None)
    rec = recall_score(trueVec, predVec, average='macro')
    f_score = 2*acc*rec/(acc+rec)
    print("整体的正确率为：%f" % acc)
    print("整体的召回率为：%f" % rec)
    print("整体的F-score为：%f" % f_score)
    prec_recall_f1score(cm)
    print("==============================================")
    showConfusionMatrix(cm)


