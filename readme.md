### 实验要求

1. 文本类别数：>=10类。
2. 训练集文档数：>=500000篇；每类平均50000篇。
3. 测试机文档数：>=500000篇；每类平均50000篇。

### 实验内容

利用朴素贝叶斯算法实现对文本的数据挖掘，主要包括：

1. 语料库的构建，主要包括利用爬虫收集Web文档等。
2. 语料库的数据预处理，包括文档建模，如去噪，分词，建立数据字典。
3. 自行实现朴素贝叶斯，训练文本分类器。
4. 对测试集的文本进行分类
5. 对测试集的分类结果利用正确率和召回率进行分析评价。

### 效果展示

1. 部分原始爬取数据

![](https://raw.githubusercontent.com/Yeefine/picBed/master/20201130225056.png)

2. 对爬取数据进行分词、去除停用词

![](https://raw.githubusercontent.com/Yeefine/picBed/master/20201130225156.png)

3. 经过TF_IDF处理

![](https://raw.githubusercontent.com/Yeefine/picBed/master/20201130225248.png)

4. 50w条测试集的测试结果

<img src="https://raw.githubusercontent.com/Yeefine/picBed/master/20201130225901.png" style="zoom: 50%;" />

<img src="https://raw.githubusercontent.com/Yeefine/picBed/master/20201130225937.png" style="zoom: 67%;" />