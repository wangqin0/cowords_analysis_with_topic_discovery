import jieba
from jieba import posseg
import re
import csv
import math
import time
import pickle
from gensim import corpora, models
import os
import sys


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

    def describe(self):
        """
        output numbers of valid words, edges, and docs
        :return: void
        """
        print("[Summary of the WordNet]")
        print('number of word_net.docs: ' + str(len(self.docs)))
        print('number of word_net.nodes: ' + str(len(self.nodes)))
        edge_counter = 0
        for node in self.edges.keys():
            for neighbor in self.edges[node]:
                edge_counter += len(neighbor)
        print('number of word_net.edges: ' + str(edge_counter))

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
            if index % 100 == 0:
                display_progress('add cut corpus', index/num_of_docs)

        for node in self.nodes:
            node.inverse_document_frequency = math.log(num_of_docs/node.doc_count + 1)

        for doc in self.docs:
            for word_id in doc.word_id_count_in_doc.keys():
                doc.word_id_tf[word_id] = doc.word_id_count_in_doc[word_id]/doc.number_of_words
                doc.word_id_tf_idf[word_id] = doc.word_id_tf[word_id] * self.nodes[word_id].inverse_document_frequency

    def word_to_id(self, word):
        return self.get_node_by_str[word].node_id

    def node_id_to_word(self, node_id):
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
            new_node = WordNode(word=unique_word, node_id=index)
            self.nodes.append(new_node)
            self.get_node_by_str[unique_word] = new_node

            self.edges[index] = {}

            index += 1
            if index % 1000 == 0:
                display_progress('generate_nodes_hash', index/num_of_nodes)

        coded_corpus = []
        for doc in cut_corpus:
            id_doc = []
            for word in doc:
                id_doc.append(self.word_to_id(word))
            coded_corpus.append(id_doc)
        return coded_corpus


class Doc:
    def __init__(self, doc_id, word_id_count_in_doc, number_of_words):
        self.id = doc_id
        self.number_of_words = number_of_words
        self.word_id_count_in_doc = word_id_count_in_doc
        self.word_id_tf = {}
        self.word_id_tf_idf = {}


class WordNode:
    def __init__(self, word, node_id=-1, word_count=0, doc_count=0, inverse_document_frequency=-1):
        self.name = word
        self.node_id = node_id
        self.word_count = word_count
        self.doc_count = doc_count
        self.inverse_document_frequency = inverse_document_frequency

    def __str__(self):
        return 'info of node "'  + str(self.node_id) \
               + '\n\tnode_name' + str(self.name) \
               + '\n\tdoc_count: ' + str(self.doc_count) \
               + '"\n\tword_count: ' + str(self.word_count) \
               + '\n\tinverse_document_frequency: ' + str(self.inverse_document_frequency)


def word_cut(list_of_docs, stop_words_set, selected_part_of_speech=None, if_output_tokens=False):
    """
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
    # selected_words_set = collect_words_to_set_from_dir(os.path.join('data', 'selected_words'))
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
            if word not in stop_words_set and part_of_speech in selected_part_of_speech and len(word) > 1:
                list_of_cut_words.append(word)
        corpus_after_cut.append(list_of_cut_words)

        index += 1
        if index % 100 == 0:
            display_progress('word cut', index/num_of_docs)

    if if_output_tokens:
        save_obj(corpus_after_cut, 'list_of_docs')

    return corpus_after_cut


def collect_words_to_set_from_dir(dir_of_txt_files):
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


def display_progress(prompt, percent):
    sys.stdout.write('\r')
    sys.stdout.write('%s [%s%s]%3.1f%s' % (
        prompt, '█' * int(percent * 50), ' ' * int(50 - int(percent * 50)), percent * 100, '%'))
    if percent == 1:
        sys.stdout.write('\n')
    sys.stdout.flush()


def save_obj(obj, name):
    with open('objects/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)