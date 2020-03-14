import catd
import os

# word_net = catd.WordNet()
word_net = catd.load_obj('reduced_tianya_posts_test_set_1000')

print(word_net.get_cut_corpus())
print()