import jieba
from jieba import posseg
import re
import pandas as pd
import math
import time
import pickle
from gensim import corpora, models


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
        extracted_words_by_tf_idf = []
        for i in range(len(self.docs)):
            sorted_list = sorted(self.docs[i].word_tf_idf, key=lambda j: self.docs[i].word_tf_idf[j], reverse=True)
            extracted_sorted_list = sorted_list[0:int(len(sorted_list) * extract_range)]
            print("doc index " + str(i) + " tf-idf extraction results: "
                  + str(len(extracted_sorted_list))
                  + "\twords extracted out of " + str(len(self.docs[i].word_tf_idf)) + " total")

            for j in sorted_list:
                print("\t" + j + "\t" + str(self.docs[i].word_tf_idf[j]))

            extracted_words_by_tf_idf += extracted_sorted_list
            print("doc index " + str(i) + "tf-idf sorted: ")
            for j in extracted_sorted_list:
                print("\t" + j + "\t" + str(self.docs[i].word_tf_idf[j]))
            print()
        return set(extracted_words_by_tf_idf)

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


def data_to_post(filename, if_output_docs=False):
    """
    :param filename:
    :param if_output_docs:
    :return: a list of documents (posts)
    """
    # excel
    # post_data_df = pd.read_excel('./data/tianya_data.xls')
    # print("[reading csv data] completed")

    # txt
    post_data_df = pd.read_csv(filename, sep="\n", header=None, error_bad_lines=False, encoding='utf-8')
    # print(post_data_df)

    combined_docs = ''
    separate_docs = []
    num_docs = 0
    for index, row in post_data_df.iterrows():
        str_1 = keep_only_chinese(str(row[0]))
        if str_1:
            num_docs += 1
            combined_docs += str_1
            separate_docs.append(str_1)
        if index % 1000 == 0:
            print(index)
    print("[generate documents] completed")

    # save docs on request
    if if_output_docs:
        save_obj(separate_docs, 'separate_docs')
        file = open('doc/combined_doc.txt', 'w', encoding='utf-8')
        file.write(str(combined_docs))
        file.close()
        file = open('doc/separate_doc.txt', 'w', encoding='utf-8')
        for token in separate_docs:
            file.write(str(token))
            file.write('\n')
        file.close()
        print("[output documents] completed")

    # print summary
    print("[data to docs] summary")
    print("\tnumber of total docs: " + str(len(post_data_df)))
    print("\tnumber of valid docs: " + str(num_docs) + "\n")

    return separate_docs


def post_to_token(doc_list, if_output_tokens=False):
    '''
    :param doc_list:
    :param if_output_tokens:
    :return: list of list, tokens grouped by documents
    '''
    stop_words_list = []
    update_stop_words_list('data/stop_words/stop_words_from_HIT.txt', stop_words_list)
    update_stop_words_list('data/stop_words/stop_words_from_baidu.txt', stop_words_list)
    print("[generate stopwords dict] complete")

    token_list = []
    # selected_flag = {'v', 'n', 'ns', 'vn', 'nt', 'nr', 's', 'nz', 'nrt', 'nrtg', 'token', 'an'}
    selected_flag = {'an', 'n', 'nr', 'ns', 'nt', 'nz', 'vn'}
    doc_list_len = len(doc_list)
    count = 1
    # selected_tokens = load_obj('selected_tokens')
    for doc in doc_list:
        curr_list = []
        # word_process_selected(doc, curr_list, stop_words_list, selected_flag, selected_tokens)
        word_process(doc, curr_list, stop_words_list, selected_flag)
        token_list.append(curr_list)

        count += 1
        if count % 1000 == 0:
            print("[generate tokens] progress: " + str(count) + "\t / " + str(doc_list_len))
    print("[generate tokens] completed")
    # pprint.pprint(token_list)

    if if_output_tokens:
        save_obj(token_list, "token_list")
        print("[output tokens] completed")

    print("[post_to_token] summary")
    print("\tnumber of stopwords: " + str(len(stop_words_list)))
    print()
    return token_list


def keep_only_chinese(curr_word):
    rule = re.compile(r"[^\u4e00-\u9fa5]")
    line = rule.sub(' ', curr_word)
    return line


def word_process_selected(curr_post, token_list, stop_words, selected_flags, selected_tokens):
    seg_counter = 0
    for word in jieba.posseg.dt.cut(curr_post):
        seg_counter += 1
        word_0, flag = str(word).split('/')
        # if current word_with_property is not a stop word_with_property and its word_with_property class is selected,
        # append it to the token_list of current doc
        if word_0 not in stop_words and word_0 in selected_tokens and flag in selected_flags and len(word_0) > 1:
            token_list.append(word_0)


def word_process(curr_post, token_list, stop_words, selected_flags):
    seg_counter = 0
    for word in jieba.posseg.dt.cut(curr_post):
        seg_counter += 1
        word_0, flag = str(word).split('/')
        # if current word_with_property is not a stop word_with_property and its word_with_property class is selected,
        # append it to the token_list of current doc
        if word_0 not in stop_words and flag in selected_flags and len(word_0) > 1:
            token_list.append(word_0)


def update_stop_words_list(file_name, stop_words_list):
    stop_words_in_file = [line.rstrip('\n') for line in open(file_name, encoding='utf-8')]
    for stop_word in stop_words_in_file:
        if stop_word not in stop_words_list:
            stop_words_list.append(stop_word)


def fresh_run():
    # Read posts from a spread sheet
    docs = data_to_post('data/original_data/tianya_posts_test_set_100.txt', if_output_docs=False)
    token_list = post_to_token(docs, if_output_tokens=True)
    word_net = WordNet()
    word_net.add_corpus(token_list)
    # for token_grouped_by_doc in word_net.nodes.keys():
    #     print(word_net.nodes[token_grouped_by_doc])
    word_net.describe()
    save_obj(word_net, 'wordnet')


def run_from_token_list():
    token_list = load_obj('token_list')
    word_net = WordNet()
    word_net.add_corpus(token_list)
    # for token_grouped_by_doc in word_net.nodes.keys():
    #     print(word_net.nodes[token_grouped_by_doc])
    word_net.describe()
    save_obj(word_net, 'word_with_property_net_selected_20')


def lda():
    corpus = []
    token_list = load_obj('token_list')
    stopwords_list = load_obj('stop_words')
    # 构造词典
    dictionary = corpora.Dictionary(token_list)
    # 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
    corpus = [dictionary.doc2bow(words) for words in token_list]
    # lda模型，num_topics设置主题的个数
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)
    # 打印所有主题，每个主题显示20个词
    for topic in lda.print_topics(num_words=20):
        print(topic)
    # 主题推断
    print(lda.inference(corpus))
    print(len(lda.inference(corpus)))


def main():
    word_net = load_obj('wordnet')

    add_node_index(word_net)
    generate_graph_csv_data(word_net)
    extract_tokens_batch(word_net)
    extract_top_words(word_net)

    print("saving wordnet obj...")
    save_obj(word_net, 'word_with_property_net')

    # extract_tokens(word_net)
    # tf_idf_extraction = load_obj('tf_idf_top_30%_extraction')
    # tf_extraction = load_obj('tf_top_30%_extraction')
    # print(len(tf_idf_extraction))
    # print(len(tf_extraction))
    # extract_top_words(word_net)
    print()


def add_node_index(word_net):
    index = 0
    for node in word_net.nodes.keys():
        word_net.nodes[node].term_frequency = index
        index += 1


def generate_graph_json_data(word_net):
    node_info_f = open('files/node_info.json', 'w+', encoding='utf-8')
    node_info_f.write('{\n"nodes": [\n')
    for node in word_net.nodes:
        node_info_f.write(str(word_net.nodes[node].term_frequency) + ', '
                          + node + ', '
                          + str(word_net.nodes[node].inverse_document_frequency) + ', '
                          + str(word_net.nodes[node].doc_count) + ', '
                          + str(word_net.nodes[node].word_count) + '\n')
    node_info_f.close()

    edge_info_f = open('files/edge_info.csv', 'w+', encoding='utf-8')
    edge_info_f.write('')
    for start_edge in word_net.edges.keys():
        for end_edge in word_net.edges[start_edge].keys():
            edge_info_f.write(str(word_net.nodes[start_edge].term_frequency) + ', '
                              + str(word_net.nodes[end_edge].term_frequency) + ', '
                              + str(word_net.edges[start_edge][end_edge]) + '\n')


def generate_graph_csv_data(word_net):
    node_info_f = open('files/node_info.csv', 'w+', encoding='utf-8')
    node_info_f.write('Id, word_with_property, inverse_document_frequency, doc_count, word_count\n')
    for node in word_net.nodes:
        node_info_f.write(str(word_net.nodes[node].term_frequency) + ', '
                          + node + ', '
                          + str(word_net.nodes[node].inverse_document_frequency) + ', '
                          + str(word_net.nodes[node].doc_count) + ', '
                          + str(word_net.nodes[node].word_count) + '\n')
    node_info_f.close()

    edge_info_f = open('files/edge_info.csv', 'w+', encoding='utf-8')
    edge_info_f.write('Source, Target, weight\n')
    for start_edge in word_net.edges.keys():
        for end_edge in word_net.edges[start_edge].keys():
            edge_info_f.write(str(word_net.nodes[start_edge].term_frequency) + ', '
                              + str(word_net.nodes[end_edge].term_frequency) + ', '
                              + str(word_net.edges[start_edge][end_edge]) + '\n')


def extract_top_words(word_net):
    tf_idf_extraction = word_net.extract_top_percent_words_by_tf_idf()
    tf_extraction = word_net.extract_top_percent_words_by_tf()

    f_tf = open('files/top_10%_words_by_tf.csv', 'w+', encoding='utf-8')
    for i in tf_extraction:
        f_tf.write(i + '\n')

    f_tf_idf = open('files/top_10%_words_by_tf_idf.csv', 'w+', encoding='utf-8')
    for i in tf_idf_extraction:
        f_tf_idf.write(i + '\n')


def extract_tokens_batch(word_net):
    for i in range(20):
        f = open('token_reduction_by_doc_count/tokens_docCount_more_than_' + str(i + 1) + '.csv', 'w+', encoding='utf-8')
        f.write('word_with_property, doc_count, word_count, inverse_document_frequency\n')
        for word in word_net.nodes.keys():
            if word_net.nodes[word].doc_count > i:
                f.write(word + ', '
                        + str(word_net.nodes[word].doc_count) + ', '
                        + str(word_net.nodes[word].word_count) + ', '
                        + str(word_net.nodes[word].inverse_document_frequency) + '\n')


def extract_tokens(word_net, minimal_doc_count):
    selected_tokens = set()
    for word in word_net.nodes.keys():
        if word_net.nodes[word].doc_count >= minimal_doc_count:
            selected_tokens.add(word)
    save_obj(selected_tokens, 'selected_tokens')


def save_obj(obj, name):
    with open('objects/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
