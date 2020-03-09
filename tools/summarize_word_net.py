import cata

word_net = cata.load_obj('word_net')
print(word_net.describe())
print(word_net.describe_docs())
print(word_net.describe_nodes())
print(word_net.describe_edges())
