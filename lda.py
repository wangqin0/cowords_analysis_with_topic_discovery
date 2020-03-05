# from https://blog.csdn.net/Yellow_python/article/details/83097994

from gensim import corpora, models
from dumpObj import *
import pprint

corpus = []
token_list = load_obj('token_list')

# for token_grouped_by_doc in token_list:
#     doc = ""
#     for token in token_grouped_by_doc:
#         doc += token
#         doc += " "
#     corpus.append(doc)

stopwords_list = load_obj('stop_words')

# 构造词典
dictionary = corpora.Dictionary(token_list)
# 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
corpus = [dictionary.doc2bow(words) for words in token_list]
# lda模型，num_topics设置主题的个数
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)
# 打印所有主题，每个主题显示20个词
for topic in lda.print_topics(num_words=20):
    print(topic)
# 主题推断
print(lda.inference(corpus))

print(len(lda.inference(corpus)))
