import re
import catd
import os
import sqlite3

tf_idf_top_percent = 0.2
doc_count_top_percent = 0.005

dataset = 'weibo_COVID19.db'
corpus_with_time = catd.util.read_sql_database_input(dataset)


stop_words_set = catd.util.collect_all_words_to_set_from_dir(os.path.join('data', 'stop_words'))

word_net = catd.WordNet()

cut_corpus_with_time = word_net.word_cut(corpus_with_time, stop_words_set)
coded_corpus = word_net.generate_nodes_hash_and_edge(cut_corpus_with_time)
word_net.add_cut_corpus(coded_corpus)
print(word_net.description())
catd.util.save_obj(word_net, 'original_' + dataset.split('.')[0])


tf_idf_selection = word_net.get_top_percent_words_by_tf_idf_in_each_doc(tf_idf_top_percent)
doc_count_selection = word_net.get_words_above_doc_count_percent(doc_count_top_percent)


word_net_with_selection = catd.WordNet()

word_net_with_selection.add_selected_word_ids_to_set(tf_idf_selection, intersection_mode=True)
word_net_with_selection.add_selected_word_ids_to_set(doc_count_selection, intersection_mode=True)

# add user_selected_words_mode parameter
cut_corpus_new = word_net_with_selection.word_cut(corpus_with_time, stop_words_set, user_selected_words_mode=True)
coded_corpus = word_net_with_selection.generate_nodes_hash_and_edge(cut_corpus_new)
word_net_with_selection.add_cut_corpus(coded_corpus)
print(word_net_with_selection.description())

# running lda
word_net_with_selection.train_lda_model()
word_net_with_selection.generate_topics_from_lda_model()

# vis
word_net_with_selection.generate_topic_graph()
word_net_with_selection.topic_time_statistics_aggregated_visualization()
word_net_with_selection.show_word_cloud()

catd.util.save_obj(word_net_with_selection, 'reduced_' + dataset.split('.')[0])
print()
