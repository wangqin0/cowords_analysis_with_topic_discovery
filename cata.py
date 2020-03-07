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


class WordNet:
    def __init__(self):
        self.nodes = []
        self.edges = {}
        self.docs = []
        self.get_node_by_str = {}


def word_cut(list_of_docs, selected_part_of_speech=None, if_output_tokens=False):
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
    stop_words_set = collect_words_to_set_from_dir(os.path.join('data', 'stop_words'))
    keep_Chinese_chars = re.compile(r"[^\u4e00-\u9fa5]")
    num_of_docs = len(list_of_docs)
    i = 0

    for doc in list_of_docs:
        list_of_cut_words = []
        # keep only Chinese characters
        doc = keep_Chinese_chars.sub(' ', doc)
        for word_with_part_of_speech in jieba.posseg.dt.cut(doc):
            word, part_of_speech = str(word_with_part_of_speech).split('/')
            if word not in stop_words_set and part_of_speech in selected_part_of_speech and len(word) > 1:
                list_of_cut_words.append(word)
        corpus_after_cut.append(list_of_cut_words)

        i += 1
        display_progress('word cut', i/num_of_docs)

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