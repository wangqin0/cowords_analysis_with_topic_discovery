import catd
import os
from shutil import copyfile

dataset = 'tianya_posts_test_set_3.txt'
tf_idf_top_percent = 0.1
doc_count_top_percent = 0.005

corpus = []
with open(os.path.join('data', 'original_data', dataset), encoding='utf-8') as f:
    for line in f:
        corpus.append(line)

stop_words_set = catd.util.collect_all_words_to_set_from_dir(os.path.join('data', 'stop_words'))

word_net = catd.WordNet()

cut_corpus = catd.util.word_cut(corpus, stop_words_set)
coded_corpus = word_net.generate_nodes_hash_and_edge(cut_corpus)
word_net.add_cut_corpus(coded_corpus)
print(word_net.description())
catd.util.save_obj(word_net, 'original_' + dataset.split('.')[0])

# keep /output/selected_words/ clean before proceed
word_net.output_top_percent_words_by_tf_idf_in_each_doc(tf_idf_top_percent)
word_net.output_words_above_doc_count_percent(doc_count_top_percent)
copyfile(os.path.join('output', 'extracted_words', 'tf_idf_top_' + str(tf_idf_top_percent) + '_words.txt'),
         os.path.join('data', 'selected_words', 'tf_idf_top_' + str(tf_idf_top_percent) + '_words.txt'))
copyfile(os.path.join('output', 'extracted_words', 'doc_count_top_' + str(doc_count_top_percent) + '_words.txt'),
         os.path.join('data', 'selected_words', 'doc_count_top_' + str(doc_count_top_percent) + '_words.txt'))

word_net_with_selection = catd.WordNet()
# rebuild word net
selected_words_set = catd.util.collect_mutual_words_to_set_from_dir(
    os.path.join('data', 'selected_words'))
# add selected_words_set parameter
cut_corpus_new = catd.util.word_cut(corpus, stop_words_set, selected_words_set)
coded_corpus = word_net_with_selection.generate_nodes_hash_and_edge(cut_corpus_new)
word_net_with_selection.add_cut_corpus(coded_corpus)
print(word_net_with_selection.description())

catd.util.save_obj(word_net_with_selection, 'reduced_' + dataset.split('.')[0])
print()
