from dumpObj import *
import math
import time


class WordNet:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.docs = {}

    def add_corpus(self, list_of_docs):
        """
        add a corpus to the WordNet, the corpus consist of a list of lists of segmented words
        does not support increment operation
        :param list_of_docs: a list of lists of segmented words
        :return: void
        """
        # building edges
        list_of_docs_len = len(list_of_docs)
        id_counter = 0
        start_time = time.time()
        print("[add corpus] processing docs")

        for doc in list_of_docs:
            word_counter_for_separate_doc = {}

            # calculate count of each word_with_property
            for word in doc:
                if word not in word_counter_for_separate_doc.keys():
                    word_counter_for_separate_doc[word] = 1
                else:
                    word_counter_for_separate_doc[word] += 1

            # calc doc_count
            for curr_node_word in word_counter_for_separate_doc:
                # create a WordNode by current word_with_property and add it to WordNet.nodes dict
                if curr_node_word not in self.nodes.keys():
                    self.nodes[curr_node_word] = WordNode(curr_node_word, doc_count=1)
                else:
                    self.nodes[curr_node_word].doc_count += 1

                # initialize a dict for current word_with_property in WordNet.edges
                if curr_node_word not in self.edges.keys():
                    self.edges[curr_node_word] = {}

                # calculate edge weight by number of co-docs
                for word_to_add in word_counter_for_separate_doc:
                    if word_to_add in self.edges[curr_node_word].keys():
                        self.edges[curr_node_word][word_to_add] += 1
                    else:
                        self.edges[curr_node_word][word_to_add] = 1

            # calc word_count
            for word in word_counter_for_separate_doc:
                self.nodes[word].word_count += word_counter_for_separate_doc[word]

            # self.docs_list.append(doc)
            self.docs[id_counter] = Doc(doc_id=id_counter, word_freq=word_counter_for_separate_doc, length=len(doc))

            # for output process status
            id_counter += 1
            if id_counter % 100 == 0:
                end_time = time.time()
                print(str(id_counter) + " / " + str(list_of_docs_len) + "\t " + str(end_time - start_time))
                start_time = time.time()

        # calc idf for each word_with_property
        for i in self.nodes.keys():
            self.nodes[i].inverse_document_frequency = math.log(len(self.docs)/self.nodes[i].doc_count + 1)

        # calc tf-idf for each word in each doc
        for i in self.docs.keys():
            for j in self.docs[i].word_tf.keys():
                self.docs[i].word_tf_idf[j] = self.docs[i].word_tf[j] * self.nodes[j].inverse_document_frequency
        print('[add corpus] complete')

    def describe(self):
        """
        output numbers of valid words, edges, and docs
        :return: void
        """
        print("[Summary of the WordNet]")
        print('len of word_net.nodes: ' + str(len(self.nodes)))
        edge_counter = 0
        for node in self.edges.keys():
            for neighbor in self.edges[node]:
                edge_counter += len(neighbor)
        print('len of word_net.edges: ' + str(edge_counter))
        print('len of word_net.docs: ' + str(len(self.docs)))

    def extract_top_percent_words_by_tf_idf(self, extract_range=0.1):
        extracted_words = []
        for i in range(len(self.docs)):
            sorted_list = sorted(self.docs[i].word_tf_idf, key=lambda j: self.docs[i].word_tf_idf[j], reverse=True)
            extracted_sorted_list = sorted_list[0:int(len(sorted_list) * extract_range)]
            print("doc index " + str(i) + " tf-idf extraction results: "
                  + str(len(extracted_sorted_list))
                  + "\twords extracted out of " + str(len(self.docs[i].word_tf_idf)) + " total")

            for j in sorted_list:
                print("\t" + j + "\t" + str(self.docs[i].word_tf_idf[j]))

            extracted_words += extracted_sorted_list
            print("doc index " + str(i) + "tf-idf sorted: ")
            for j in extracted_sorted_list:
                print("\t" + j + "\t" + str(self.docs[i].word_tf_idf[j]))
            print()
        return set(extracted_words)

    def extract_top_percent_words_by_tf(self, extract_range=0.1):
        extracted_words = []
        for i in range(len(self.docs)):
            sorted_list = sorted(self.docs[i].word_tf, key=lambda j: self.docs[i].word_tf[j], reverse=True)
            extracted_sorted_list = sorted_list[0:int(len(sorted_list) * extract_range)]
            print("doc index " + str(i) + " tf-idf extraction results: "
                  + str(len(extracted_sorted_list))
                  + "\twords extracted out of " + str(len(self.docs[i].word_tf)) + " total")

            for j in sorted_list:
                print("\t" + j + "\t" + str(self.docs[i].word_tf[j]))

            extracted_words += extracted_sorted_list
            print("doc index " + str(i) + "tf-idf sorted: ")
            for j in extracted_sorted_list:
                print("\t" + j + "\t" + str(self.docs[i].word_tf[j]))
            print()
        return set(extracted_words)


class WordNode:
    def __init__(self, name, word_count=0, doc_count=0, id=-1, inverse_document_frequency=-1):
        self.name = name
        self.id = id
        self.word_count = word_count
        self.doc_count = doc_count
        self.inverse_document_frequency = inverse_document_frequency

    def __str__(self):
        return 'info of node "' + str(self.name) \
               + '"\n\tword_count: ' + str(self.word_count) \
               + '\n\tdoc_count: ' + str(self.doc_count) \
               + '\n\tid(*): ' + str(self.id) \
               + '\n\tinverse_document_frequency: ' + str(self.inverse_document_frequency)


class Doc:
    def __init__(self, doc_id, word_freq=None, length=-1):
        self.id = doc_id
        self.len = length
        self.word_count_in_doc = word_freq
        self.word_tf = {}
        self.word_tf_idf = {}
        for word in word_freq:
            self.word_tf[word] = word_freq[word]/self.len

    def __str__(self):
        return 'info of doc ' + self.id + '\n\tlen: ' + self.len
