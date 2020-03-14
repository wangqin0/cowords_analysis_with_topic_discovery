# should take less than 2 minute to run all the tests

import catd
import os

corpus = []
with open(os.path.join('data', 'original_data', 'tianya_posts_test_set_100.txt'), encoding='utf-8') as f:
    for line in f:
        corpus.append(line)

stop_words_set = catd.collect_all_words_to_set_from_dir(os.path.join('data', 'stop_words'))

cut_corpus = catd.word_cut(corpus, stop_words_set)

word_net = catd.WordNet()
coded_corpus = word_net.generate_nodes_hash_and_edge(cut_corpus)
word_net.add_cut_corpus(coded_corpus)
catd.save_obj(word_net, 'word_net')
print()
