import catd

word_net = catd.util.load_obj('weibo_COVID19_reduced_0.27_0.02_intersection')
# word_net = catd.WordNet()

print(word_net.description())
