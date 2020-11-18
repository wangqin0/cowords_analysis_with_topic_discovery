# should take less than 2 minute to run all the tests

import catd
import os

catd.util.set_up_current_dir_as_working_dir()

dataset = 'test_weibo_COVID19.db'
corpus_with_time = catd.util.get_sql_database_input(dataset)

stop_words_set = catd.util.collect_all_words_to_set_from_dir(os.path.join('data', 'stop_words'))

word_net = catd.WordNet()

cut_corpus_with_time = word_net.word_cut(corpus_with_time, stop_words_set)
coded_corpus = word_net.generate_nodes_hash_and_edge(cut_corpus_with_time)
word_net.add_cut_corpus(coded_corpus)

print(word_net.description())

# break point here to see what's inside
catd.util.save_obj(word_net, 'original_' + dataset.split('.')[0])

