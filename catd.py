import math
import os
import pickle
import re
import sys
from wordcloud import WordCloud
from gensim import corpora, models
import jieba
from jieba import posseg


def count_word_in_doc(list_of_words):
    word_counter_for_separate_doc = {}
    for word in list_of_words:
        if word not in word_counter_for_separate_doc.keys():
            word_counter_for_separate_doc[word] = 1
        else:
            word_counter_for_separate_doc[word] += 1
    return word_counter_for_separate_doc


class WordNet:
    def __init__(self):
        self.docs = []
        self.nodes = []
        self.edges = {}
        self.get_node_by_str = {}

    def description(self):
        """
        output numbers of valid words, edges, and docs
        :return: void
        """
        num_of_edges = 0
        for word_id in self.edges.keys():
            num_of_edges += len(self.edges[word_id])
        return '[word_net info]\n' \
               '\tnumber of word_net.docs: {}\n\tnumber of word_net.nodes: {}\n\tnumber of word_net.edges: {}'\
            .format(len(self.docs), len(self.nodes), num_of_edges)

    def docs_description(self):
        docs_info = '[docs info]'
        for doc in self.docs:
            docs_info += doc.description(self)
        return docs_info

    def nodes_description(self):
        nodes_info = '[nodes info]\n'
        for node in self.nodes:
            nodes_info += str(node) + '\n'
        return nodes_info

    def edges_description(self):
        edge_info = '[edge info]'
        for word_id in self.edges.keys():
            for neighbor_id in self.edges[word_id]:
                edge_info += '\t{:8}'.format(self.word_id_to_word(word_id)) \
                               + '-> {:8}'.format(self.word_id_to_word(neighbor_id)) \
                               + '  {}'.format(self.edges[word_id][neighbor_id]) + '\n'
        return edge_info

    def add_cut_corpus(self, coded_corpus):
        """
        :param coded_corpus: list of lists of cut words
        :return: void
        """

        num_of_docs = len(coded_corpus)
        index = 0

        # generate nodes
        for doc in coded_corpus:
            word_id_counter = count_word_in_doc(doc)

            for node_id in word_id_counter.keys():
                # update doc_count and word_count
                self.nodes[node_id].doc_count += 1
                self.nodes[node_id].word_count += word_id_counter[node_id]

                for neighbor_id in word_id_counter:
                    if neighbor_id not in self.edges[node_id].keys():
                        self.edges[node_id][neighbor_id] = 1
                    else:
                        self.edges[node_id][neighbor_id] += 1

            self.docs.append(Doc(doc_id=index, word_id_count_in_doc=word_id_counter, number_of_words=len(doc)))

            index += 1
            display_progress('add cut corpus', index, num_of_docs)

        for node in self.nodes:
            node.inverse_document_frequency = math.log(num_of_docs / node.doc_count + 1)

        for doc in self.docs:
            for word_id in doc.word_id_count_in_doc.keys():
                doc.word_id_tf[word_id] = doc.word_id_count_in_doc[word_id] / doc.number_of_words
                doc.word_id_tf_idf[word_id] = doc.word_id_tf[word_id] * self.nodes[word_id].inverse_document_frequency

    def get_cut_corpus(self):
        corpus = []
        for doc in self.docs:
            word_list = []
            for word_id in doc.word_id_count_in_doc.keys():
                for i in range(doc.word_id_count_in_doc[word_id]):
                    word_list.append(self.word_id_to_word(word_id))
            corpus.append(word_list)
        return corpus

    def extract_topics(self):
        corpus = self.get_cut_corpus()
        corpora

        pass

    def output_top_percent_words_by_tf_idf_in_each_doc(self, percent):
        extracted_words_id_set = set()
        for doc in self.docs:
            sorted_list = sorted(doc.word_id_tf_idf, key=lambda j: doc.word_id_tf_idf[j], reverse=True)
            extracted_sorted_list = sorted_list[0:int(len(sorted_list) * percent)]
            extracted_words_id_set.update(set(extracted_sorted_list))
        extracted_words = ''
        for word_id in extracted_words_id_set:
            extracted_words += self.word_id_to_word(word_id) + '\n'
        with open(os.path.join('output', 'extracted_words', 'tf_idf_top_' + str(percent) + '_words.txt'), 'w+', encoding='utf-8') as f:
            f.write(extracted_words)
        return extracted_words

    def output_words_above_doc_count_percent(self, percent):
        extracted_words = []
        file_content = ''
        threshold = int(len(self.docs) * percent)
        for node in self.nodes:
            if node.doc_count > threshold:
                extracted_words.append(node.word)
                file_content += node.word + '\n'

        with open(os.path.join('output', 'extracted_words', 'doc_count_top_' + str(percent) + '_words.txt'), 'w+', encoding='utf-8') as f:
            f.write(file_content)

        return extracted_words

    def word_to_id(self, word):
        return self.get_node_by_str[word].node_id

    def word_id_to_word(self, node_id):
        return self.nodes[node_id].word

    def generate_nodes_hash_and_edge(self, cut_corpus):
        """
        setup all the nodes, edges, and a dict to get nodes by word
        :param cut_corpus: list of lists of cut word
        :return: void
        """
        word_set = set()
        for doc in cut_corpus:
            for word in doc:
                word_set.add(word)

        num_of_nodes = len(word_set)
        index = 0

        for unique_word in word_set:
            new_node = WordNode(node_id=index, word=unique_word)
            self.nodes.append(new_node)
            self.get_node_by_str[unique_word] = new_node

            self.edges[index] = {}

            index += 1
            display_progress('generate_nodes_hash', index, num_of_nodes)

        coded_corpus = []
        for doc in cut_corpus:
            id_doc = []
            for word in doc:
                id_doc.append(self.word_to_id(word))
            coded_corpus.append(id_doc)
        return coded_corpus

    def export(self, dest_dir=os.path.join('output', 'exported'), optional_postfix=''):
        if len(optional_postfix) != 0:
            optional_postfix = '_' + optional_postfix
        with open(os.path.join(dest_dir, 'data_structure', optional_postfix, '.txt'),
                  'w+', encoding='utf-8') as output_file:
            output_file.write('# word_net.docs')
            for doc in self.docs:
                pass


class Doc:
    def __init__(self, doc_id, word_id_count_in_doc, number_of_words):
        self.doc_id = doc_id
        self.number_of_words = number_of_words
        self.word_id_count_in_doc = word_id_count_in_doc
        self.word_id_tf = {}
        self.word_id_tf_idf = {}

    def __str__(self):
        return '\n[Doc info]\ndoc_id: {0}\tnumber_of_words: {1}'.format(self.doc_id, self.number_of_words)
        pass

    def description(self, word_net):
        doc_info = '\n[Doc info] doc_id: {0}\tnumber_of_words: {1}\n\tword_id_count_in_doc:\n' \
                   '\t\t count |   tf   | tf_idf |  word ' \
            .format(self.doc_id, self.number_of_words)
        for word_id in self.word_id_count_in_doc.keys():
            doc_info += '\n\t\t  {0:>2}     {1:2.4f}   {2:2.4f}    {3:<4}' \
                            .format(self.word_id_count_in_doc[word_id], self.word_id_tf[word_id],
                                    self.word_id_tf_idf[word_id], word_net.nodes[word_id].word)
        return doc_info


class WordNode:
    def __init__(self, node_id, word, doc_count=0, word_count=0, inverse_document_frequency=-1):
        self.node_id = node_id
        self.word = word
        self.doc_count = doc_count
        self.word_count = word_count
        self.inverse_document_frequency = inverse_document_frequency

    def __str__(self):
        return '[node info] id: {:8} doc_count: {:5}  word_count: {:8}  inverse_document_frequency: {:3.5}  word: {}'\
                   .format(self.node_id, self.doc_count, self.word_count, self.inverse_document_frequency, self.word)


def word_cut(list_of_docs, stop_words_set, selected_words_set=None, selected_part_of_speech= None, if_output_tokens=False):
    """
    :param stop_words_set:
    :param if_output_tokens:
    :param selected_part_of_speech:
    :param list_of_docs:
    :param if_output_docs:
    :return: a list of cleaned documents contain only Chinese characters
    """
    if selected_part_of_speech is None:
        # set default selection
        selected_part_of_speech = {'an', 'n', 'nr', 'ns', 'nt', 'nz', 'vn'}

    # TODO: add word filter
    # selected_words_set = collect_all_words_to_set_from_dir(os.path.join('data', 'selected_words'))
    # and word in selected_words_set

    corpus_after_cut = []
    keep_chinese_chars = re.compile(r"[^\u4e00-\u9fa5]")
    num_of_docs = len(list_of_docs)
    index = 0

    for doc in list_of_docs:
        list_of_cut_words = []
        # keep only Chinese characters
        doc = keep_chinese_chars.sub(' ', doc)
        for word_with_part_of_speech in jieba.posseg.dt.cut(doc):
            word, part_of_speech = str(word_with_part_of_speech).split('/')
            if selected_words_set is None:
                if word not in stop_words_set and part_of_speech in selected_part_of_speech and len(word) > 1:
                    list_of_cut_words.append(word)
            else:
                if word not in stop_words_set and part_of_speech in selected_part_of_speech and len(word) > 1 and word in selected_words_set:
                    list_of_cut_words.append(word)
        corpus_after_cut.append(list_of_cut_words)

        index += 1
        display_progress('word cut', index, num_of_docs)

    if if_output_tokens:
        save_obj(corpus_after_cut, 'list_of_docs')

    return corpus_after_cut


def collect_all_words_to_set_from_dir(dir_of_txt_files):
    """
    generate stop words set from a given directory, by iterating all the txt files
    :param dir_of_txt_files: each txt file should contain words sperated by '\n'
    :return: a complete set of words
    """
    word_set = set()
    for filename in os.listdir(dir_of_txt_files):
        if filename.endswith(".txt"):
            file = os.path.join(dir_of_txt_files, filename)
            words = [line.rstrip('\n') for line in open(file, encoding='utf-8')]
            for word in words:
                word_set.add(word)
    return word_set


def collect_mutual_words_to_set_from_dir(dir_of_txt_files):
    """
    generate stop words set from a given directory, by iterating all the txt files
    :param dir_of_txt_files: each txt file should contain words sperated by '\n'
    :return: a complete set of words
    """
    word_set = set()
    is_initialized = False
    for filename in os.listdir(dir_of_txt_files):
        curr_word_set = set()
        if filename.endswith(".txt"):
            file = os.path.join(dir_of_txt_files, filename)
            words = [line.rstrip('\n') for line in open(file, encoding='utf-8')]
            for word in words:
                curr_word_set.add(word)
            if is_initialized is True and len(curr_word_set) != 0:
                word_set = word_set.intersection(curr_word_set)
            else:
                word_set = curr_word_set
                is_initialized = True
    return word_set


def display_progress(prompt, curr_progress, total):
    if curr_progress % 100 == 0:
        progress_percent = curr_progress/total
        sys.stdout.write('\r')
        sys.stdout.write('%s [%s%s]%3.1f%s' % (
            prompt, '█' * int(progress_percent * 50), ' ' * int(50 - int(progress_percent * 50)), progress_percent * 100, '%'))
    elif curr_progress == total:
        sys.stdout.write('\r')
        sys.stdout.write('%s [%s]100%s\n' % (prompt, '█' * 50, '%'))
    sys.stdout.flush()


def save_obj(obj, name):
    with open(os.path.join('output', 'objects', name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(os.path.join('output', 'objects', name + '.pkl'), 'rb') as f:
        return pickle.load(f)