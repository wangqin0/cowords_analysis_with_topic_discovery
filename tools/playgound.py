# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

import catd

n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20


# word_net = catd.load_obj('reduced_tianya_posts_test_set_1000')
# data_samples = word_net.get_cut_corpus()

with open('../data/original_data/tianya_posts_test_set_100.txt', 'r', encoding='utf-8') as f:
    data_samples = []
    for line in f:
        data_samples.append(line)

# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words=None)
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words=None)
tf = tf_vectorizer.fit_transform(data_samples)

print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))

lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)

print()
