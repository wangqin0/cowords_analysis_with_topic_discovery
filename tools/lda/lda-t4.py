from tools.dumpObj import *
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pprint
import numpy as np


np.set_printoptions(threshold=np.inf)

corpus = []
token_list = load_obj('token-list')


for token_grouped_by_doc in token_list:
    doc = ""
    for token in token_grouped_by_doc:
        doc += token
        doc += " "
    corpus.append(doc)

stpwrdlst = load_obj('stop-words')


pp = pprint.PrettyPrinter(indent=2)

cntVector = CountVectorizer(stop_words=stpwrdlst)

cntTf = cntVector.fit_transform(corpus)
# print(cntTf)

lda = LatentDirichletAllocation(n_components=10,
                                random_state=0)
print("[running LDA] complete")
docres = lda.fit_transform(cntTf)

print("\n[docres]")
print(docres)
print("\n[lda.components_]")
print(lda.components_)
