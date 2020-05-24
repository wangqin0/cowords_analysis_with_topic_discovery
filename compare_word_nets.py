import os
import catd


original_wordnet = catd.util.load_obj('original_weibo_COVID19')
original_num_of_nodes = len(original_wordnet.nodes)
original_num_of_edges = 0
print(original_wordnet.description(), '\n')

dir = os.path.join('output', 'objects')
for word_id in original_wordnet.edges.keys():
    original_num_of_edges += len(original_wordnet.edges[word_id])
counter = 0
for word_net_object in os.listdir(dir):
    if '.pkl' in word_net_object and 'weibo_COVID19_reduced_' in word_net_object:
        object_name = word_net_object.split('.pkl')[0]
        word_net = catd.util.load_obj(object_name)
        print('[' + str(counter) + ']', object_name)
        num_of_edges = 0
        for word_id in word_net.edges.keys():
            num_of_edges += len(word_net.edges[word_id])
        num_of_nodes = len(word_net.nodes)

        print('\tNodes reduced to: {}, \t\t{:.4f}%'
              '\n\tEdges reduced to: {},\t\t{:.4f}%'
              '\n\tedge to node reduction ratio: {}%\n'
              .format(num_of_nodes, num_of_nodes/original_num_of_nodes * 100,
                      num_of_edges, num_of_edges/original_num_of_edges * 100,
                      (num_of_edges/original_num_of_edges) / (num_of_nodes/original_num_of_nodes) * 100))
        counter += 1
