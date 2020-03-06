from tools.dumpObj import *


def main():
    word_net = load_obj('word_with_property_net')

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
    node_info_f = open('files/node_info.json', 'w+')
    node_info_f.write('{\n"nodes": [\n')
    for node in word_net.nodes:
        node_info_f.write(str(word_net.nodes[node].term_frequency) + ', '
                          + node + ', '
                          + str(word_net.nodes[node].inverse_document_frequency) + ', '
                          + str(word_net.nodes[node].doc_count) + ', '
                          + str(word_net.nodes[node].word_count) + '\n')
    node_info_f.close()

    edge_info_f = open('files/edge_info.csv', 'w+')
    edge_info_f.write('')
    for start_edge in word_net.edges.keys():
        for end_edge in word_net.edges[start_edge].keys():
            edge_info_f.write(str(word_net.nodes[start_edge].term_frequency) + ', '
                              + str(word_net.nodes[end_edge].term_frequency) + ', '
                              + str(word_net.edges[start_edge][end_edge]) + '\n')


def generate_graph_csv_data(word_net):
    node_info_f = open('files/node_info.csv', 'w+')
    node_info_f.write('Id, word_with_property, inverse_document_frequency, doc_count, word_count\n')
    for node in word_net.nodes:
        node_info_f.write(str(word_net.nodes[node].term_frequency) + ', '
                          + node + ', '
                          + str(word_net.nodes[node].inverse_document_frequency) + ', '
                          + str(word_net.nodes[node].doc_count) + ', '
                          + str(word_net.nodes[node].word_count) + '\n')
    node_info_f.close()

    edge_info_f = open('files/edge_info.csv', 'w+')
    edge_info_f.write('Source, Target, weight\n')
    for start_edge in word_net.edges.keys():
        for end_edge in word_net.edges[start_edge].keys():
            edge_info_f.write(str(word_net.nodes[start_edge].term_frequency) + ', '
                              + str(word_net.nodes[end_edge].term_frequency) + ', '
                              + str(word_net.edges[start_edge][end_edge]) + '\n')


def extract_top_words(word_net):
    tf_idf_extraction = word_net.extract_top_percent_words_by_tf_idf()
    tf_extraction = word_net.extract_top_percent_words_by_tf()

    f_tf = open('files/top_10%_words_by_tf.csv', 'w+')
    for i in tf_extraction:
        f_tf.write(i + '\n')

    f_tf_idf = open('files/top_10%_words_by_tf_idf.csv', 'w+')
    for i in tf_idf_extraction:
        f_tf_idf.write(i + '\n')


def extract_tokens_batch(word_net):
    for i in range(20):
        f = open('token_reduction_by_doc_count/tokens_docCount_more_than_' + str(i + 1) + '.csv', 'w+')
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


if __name__ == '__main__':
    main()