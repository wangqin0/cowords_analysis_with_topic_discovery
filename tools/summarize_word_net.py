import catd
import os

word_net_filename = 'word_net_with_selection'

word_net = catd.load_obj(word_net_filename)

with open(os.path.join('output', 'description', word_net_filename + '.txt'), 'w+', encoding='utf-8') as output_file:
    output_file.write(word_net.description())
    # output_file.write(word_net.docs_description())
    output_file.write(word_net.nodes_description())
    # output_file.write(word_net.edges_description())
