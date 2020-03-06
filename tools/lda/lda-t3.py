import jieba
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('京州', True)
# 第一个文档分词#
with open('./nlp_test0.txt') as f:
    document = f.read().encode('GBK')
    # document_decode = document.decode('GBK')
    document_cut = jieba.cut(document)
    # document_cut = jieba.cut(document_decode)
    # print  ' '.join(jieba_cut)
    result = ' '.join(document_cut)
    result = result.encode('utf-8')
    with open('nlp_test2.txt', 'w') as f2:
        f2.write(str(result))
f.close()
f2.close()

# 第二个文档分词#
with open('nlp_test4.txt') as f:
    document2 = f.read().encode('GBK')
    # document2_decode = document2.decode('GBK')
    # document2_cut = jieba.cut(document2_decode)
    document2_cut = jieba.cut(document2)
    # print  ' '.join(jieba_cut)
    result = ' '.join(document2_cut)
    result = result.encode('utf-8')
    with open('./nlp_test3.txt', 'w') as f2:
        f2.write(str(result))
f.close()
f2.close()

# 第三个文档分词
jieba.suggest_freq('桓温', True)
with open('./nlp_test4.txt') as f:
    document3 = f.read().encode('GBK')
    # document3_decode = document3.decode('GBK')
    # document3_cut = jieba.cut(document3_decode)
    document3_cut = jieba.cut(document3)
    # print  ' '.join(jieba_cut)
    result = ' '.join(document3_cut)
    result = result.encode('utf-8')
    with open('./nlp_test5.txt', 'w') as f3:
        f3.write(str(result))
f.close()
f3.close()

with open('nlp_test2.txt') as f3:
    res1 = f3.read()
print(res1)
with open('./nlp_test3.txt') as f4:
    res2 = f4.read()
print(res2)
with open('./nlp_test5.txt') as f5:
    res3 = f5.read()
print(res3)

# 从文件导入停用词表
stpwrdpath = "stop-words/哈工大停用词表.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
# 将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()





# corpus 输入为空格分隔的分词结果
print("\n[running LDA]")
corpus = [res1, res2, res3]
cntVector = CountVectorizer(stop_words=stpwrdlst)
cntTf = cntVector.fit_transform(corpus)

print(type(res1))
print(type(corpus))
print(type(stpwrdlst))
# print(cntTf)

lda = LatentDirichletAllocation(n_components=3,
                                learning_offset=50.,
                                random_state=0)
# lda = LatentDirichletAllocation(n_topics=2,
#                                 learning_offset=50.,
#                                 random_state=0)
print("[running LDA] complete")
docres = lda.fit_transform(cntTf)

print("\n[docres]")
print(docres)
print("\n[lda.components_]")
print(lda.components_)
