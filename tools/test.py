# should take less than 2 minute to run all the tests

import cata
import os

corpus = []
with open(os.path.join('data', 'original_data', 'tianya_posts_test_set_100.txt')) as f:
    for line in f:
        corpus.append(line)

token_list = cata.word_cut(corpus)

print(token_list)