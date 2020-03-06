import numpy as np
import pandas as pd

df = pd.read_excel(r"C:\Users\yxy\IPYNB\DATA\data00.xlsx", encoding='gbk')


# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


stop_words1 = stopwordslist(r'C:\Users\yxy\IPYNB\DATA\stopwords1.txt')


# 去除停用词
def remove_stopwords(text, stopwords_filepath=r'C:\Users\yxy\IPYNB\DATA\stopwords1.txt'):
    stopwords = stopwordslist(stopwords_filepath)  # 这里加载停用词的路径
    for word in text:
        word_without_stopwords = ' '
        if word not in stopwords:
            if word != '\t':
                word_without_stopwords += ' ' + word
    return word_without_stopwords


# 分词函数，精确模式
import jieba


def CUT_CHINESE_WORD(text, userdict_filepath=r"stop-words/userdict.txt",
                     stopwords_filepath=r'stop-words/哈工大停用词表.txt'):
    jieba.load_userdict(userdict_filepath)
    stopwords = stopwordslist(stopwords_filepath)  # 这里加载停用词的路径
    seg_list = jieba.cut(text, cut_all=False)  # cut_all=True 全模式
    seg_list_without_stopwords = ' '
    for word in seg_list:
        if word not in stopwords:
            if word != '\t':
                seg_list_without_stopwords += ' ' + word
    return seg_list_without_stopwords


def cut_word(text):
    cutted_word = jieba.cut(text, cut_all=False)
    return cutted_word


# 分词
t = pd.DataFrame(df['转发内容'].astype(str))
t["分词_转发内容"] = t.转发内容.apply(CUT_CHINESE_WORD)
# t.分词_转发内容.head()

# t["分词_转发内容1"]=t.转发内容.apply(cut_word)

##词性分词
# import jieba.posseg as pseg
# t["词性_分词"]=t.转发内容.apply(pseg.cut)
# t.词性_分词.head()

# def print_word(w):
#    for token_grouped_by_doc in w:
#        print ('\t',token_grouped_by_doc)


# t.词性_分词.apply(print_fenci_cixing)


# t.词性_分词[1:6].apply(print_fenci_cixing)
# 根据词性和词的长度过滤掉助词、介词、连词等虚词以及词长较短
# 按词性去词
# def cixing_quci(w):
#    for token_grouped_by_doc in w:
#        new_list=[]
#        if token_grouped_by_doc.flag!= 'x':
#            new_list.append(x)
#    return new_list
#
# t["去词_分词"]=t.词性_分词.apply(cixing_quci)

# list转str
# def list2str(list):
#    list1=list
##    str1=' '
##    for word_with_property in list:
##        str1  +=' '+word_with_property
#    return ','.join(list1)

# 用结巴分词工具寻找关键词
# from jieba import analyse
# tfidf=analyse.extract_tags
# t["关键词1"]=t.转发内容.apply(tfidf)

# t["关键词2"]=[str(token_grouped_by_doc) for token_grouped_by_doc in t.关键词1]
# t.关键词2.apply(remove_stopwords)
# t["关键词3"]=t.关键词2.apply(remove_stopwords)


# t.关键词.head()
# t["关键词2"]=t.转发内容.apply(tfidf)
# t["关键词3"]=t.转发内容.apply(tfidf)
# t["关键词4"]=t.转发内容.apply(tfidf)


# t.关键词[1:10].apply(print_word)


# 分类LDA：随机的方法，每一次结果都不一样
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

n_features = 2500
tf0_vectorizer = CountVectorizer(strip_accents='unicode',
                                 max_features=n_features,
                                 stop_words='english',
                                 max_df=0.5,
                                 min_df=10)

tf0 = tf0_vectorizer.fit_transform(t.分词_转发内容)
# test0=tf0_vectorizer.fit_transform(t.分词_转发内容[900:1000])


weight0 = tf0_vectorizer.fit_transform(t.分词_转发内容).toarray()

# LDA
from sklearn.decomposition import LatentDirichletAllocation

n_topics = 10
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                learning_method='online',
                                learning_offset=50,
                                random_state=0)

lda.fit(tf0)


def print_top_words(model, feature_names, n_top_words):  # features_names:list
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:n_top_words - 1:-1]]))
    print()


n_top_words = 20

tf0_feature_names = tf0_vectorizer.get_feature_names()
print_top_words(lda, tf0_feature_names, n_top_words)

docres0 = lda.fit_transform(tf0)  # 内容的主题分布
perplexity0 = lda.perplexity(tf0)  # 训练的收敛值

# doc_topic_dist_unnormalized = np.matrix(lda.transform(test0))
# normalize the distribution (only needed if you want to work with the probabilities)
# doc_topic_dist = doc_topic_dist_unnormalized/doc_topic_dist_unnormalized.sum(axis=1)


# tf1_feature_names=tf_vectorizer.get_feature_names()
# print_top_words(lda,tf1_feature_names,n_top_words)

# 下一步，对新内容进行分词，归类，确定主题向量
# """ 需要每个类的分词列表，
#      但是上面分类的词保存在哪里，
#      何种类型，
#      如何调用
#      """
#
# def to_features(model,new_weibo,t):#features:特征词，new_weibo:新内容 t为归类阈值
#    fenci__weibo=CUT_CHINESE_WORD(new_weibo)
#    topic_features=[]
#    topic_vector=[]
#    k=0
#    while k<10:
#        for topic_idx,topic in enumerate(model.components_):
#              topic_features[k]=[topic[token]
#                    for token in topic.argsort()[:n_top_words -1:-1]]
#    k+=1
#
#    while token_grouped_by_doc<10:
#        for word_with_property in fenci_new_weibo:
#            weight=0#需要一个数组，存放new_weibo在各主题下的权重
#            k=0
#            while k<10:
#                for features in topic_features[k]:
#                    if word_with_property in topic_features[k]:
#                        weight+=1
#                if weight>=t:
#                    topic_vector[token_grouped_by_doc]=1
#            k+=1
#    token_grouped_by_doc+=1
#
#    return topic_vector
#
# weibo="触不及防我被撩了酥死我了！啊啊啊啊啊啊我的天！！！#二次元#L史上第一最腐女的秒拍视频​​​​小窗口"
# top_vec_weibo=to_features(lda,weibo,20)
# print (top_vec_weibo)
#


# 测试集，训练集
