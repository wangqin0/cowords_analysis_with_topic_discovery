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


def construct_word_net(corpus_with_time, stop_words_set, dataset):
    word_net = catd.WordNet()
    cut_corpus_with_time = word_net.word_cut(corpus_with_time, stop_words_set)
    coded_corpus = word_net.generate_nodes_hash_and_edge(cut_corpus_with_time)
    word_net.add_cut_corpus(coded_corpus)
    catd.util.save_obj(word_net, 'original_' + dataset.split('.')[0])
    return word_net


def reduce_word_net(original_word_net, dataset, corpus_with_time, stop_words_set, tf_idf_top_percent, doc_count_top_percent, is_intersection):
    tf_idf_selection = original_word_net.get_top_percent_words_by_tf_idf_in_each_doc(tf_idf_top_percent)
    doc_count_selection = original_word_net.get_words_above_doc_count_percent(doc_count_top_percent)
    word_net_with_selection = catd.WordNet()
    word_net_with_selection.add_selected_word_ids_to_set(tf_idf_selection, intersection_mode=is_intersection)
    word_net_with_selection.add_selected_word_ids_to_set(doc_count_selection, intersection_mode=is_intersection)
    # add user_selected_words_mode parameter
    cut_corpus_new = word_net_with_selection.word_cut(corpus_with_time, stop_words_set, user_selected_words_mode=True)
    coded_corpus = word_net_with_selection.generate_nodes_hash_and_edge(cut_corpus_new)
    word_net_with_selection.add_cut_corpus(coded_corpus)

    # running lda
    # if len(word_net_with_selection.nodes) == 0 or len(word_net_with_selection.edges) == 0:
    #     print('    Empty net, skip\n')
    #     catd.util.save_obj(word_net_with_selection,
    #                        dataset.split('.')[0]
    #                        + '_reduced_' + str(tf_idf_top_percent) + '_' + str(doc_count_top_percent)
    #                        + ('intersection' if is_intersection else '_non_intersection'))
    #     return word_net_with_selection
    #
    # word_net_with_selection.train_lda_model()
    # word_net_with_selection.generate_topics_from_lda_model()
    # catd.util.save_obj(word_net_with_selection,
    #                    dataset.split('.')[0]
    #                    + '_reduced_' + str(tf_idf_top_percent) + '_' + str(doc_count_top_percent)
    #                    + ('intersection' if is_intersection else '_non_intersection'))

    catd.util.save_obj(word_net_with_selection,
                       dataset.split('.')[0]
                       + '_reduced_' + str(tf_idf_top_percent) + '_' + str(doc_count_top_percent)
                       + ('_intersection' if is_intersection else '_non_intersection'))
    return word_net_with_selection


def main():
    # tf_idf_top_percent = 0.2
    # doc_count_top_percent = 0.005
    # is_intersection = True

    dataset = 'weibo_COVID19.db'
    corpus_with_time = read_sql_database_input(dataset)
    stop_words_set = catd.util.collect_all_words_to_set_from_dir(os.path.join('data', 'stop_words'))

    print('Constructing original WordNet.')
    original_word_net = construct_word_net(corpus_with_time, stop_words_set, dataset)
    # original_word_net = catd.util.load_obj('original_weibo_COVID19')
    print(original_word_net.description())

    is_intersection = True
    for tf_idf_top_percent in (i / 100 for i in range(0, 30, 3)):
        for doc_count_top_percent in (i / 1000 for i in range(0, 50, 5)):
            print('[Reduce original word_net] '
                  '\n\ttf_idf_top_percent = {}, '
                  '\n\tdoc_count_top_percent = {}, '
                  '\n\tis_intersection = {}'
                  .format(tf_idf_top_percent, doc_count_top_percent, is_intersection))
            reduced_word_net = reduce_word_net(original_word_net,
                                               dataset,
                                               corpus_with_time,
                                               stop_words_set,
                                               tf_idf_top_percent,
                                               doc_count_top_percent,
                                               is_intersection)
            print(reduced_word_net.description())


if __name__ == '__main__':
    main()
