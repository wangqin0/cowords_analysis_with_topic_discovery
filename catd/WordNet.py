import os
import math
from .WordNode import WordNode
from .Doc import Doc
from .util import *
from gensim import corpora, models


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
        with open(os.path.join('../output', 'extracted_words', 'tf_idf_top_' + str(percent) + '_words.txt'), 'w+', encoding='utf-8') as f:
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

        with open(os.path.join('../output', 'extracted_words', 'doc_count_top_' + str(percent) + '_words.txt'), 'w+', encoding='utf-8') as f:
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

    def export(self, dest_dir=os.path.join('../output', 'exported'), optional_postfix=''):
        if len(optional_postfix) != 0:
            optional_postfix = '_' + optional_postfix
        with open(os.path.join(dest_dir, 'data_structure', optional_postfix, '.txt'),
                  'w+', encoding='utf-8') as output_file:
            output_file.write('# word_net.docs')
            for doc in self.docs:
                pass
