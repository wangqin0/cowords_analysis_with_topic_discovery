import re
import catd
import os
import sqlite3


def read_txt_input(dataset_filename):
    corpus = []
    with open(os.path.join('data', 'original_data', dataset_filename), encoding='utf-8') as f:
        for line in f:
            corpus.append(line)
    read_txt_input(corpus)


def read_sql_database_input(database_filename):
    remove_hashtag = re.compile(r'#[\w-]+#')
    con = sqlite3.connect(os.path.join('data', 'original_data', database_filename))
    cursor = con.cursor()
    cursor.execute("SELECT post_content, post_time FROM posts")
    rows = cursor.fetchall()
    corpus = [(remove_hashtag.sub(' ', str(post_content)), post_time) for post_content, post_time in rows]
    return corpus


# tf_idf_top_percent = 0.1
tf_idf_top_percent = 0.2
doc_count_top_percent = 0.005
# dataset = 'tianya_posts_test_set_300.txt'
# corpus_with_time = read_txt_input(dataset)

dataset = 'weibo_COVID19_complete.db'
corpus_with_time = read_sql_database_input(dataset)


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
word_net_with_selection.generate_lda_model()
topics = word_net_with_selection.get_topics()

catd.util.save_obj(word_net_with_selection, 'reduced_' + dataset.split('.')[0])
print()
