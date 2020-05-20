import catd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

word_net = catd.util.load_obj('reduced_weibo_COVID19')
# word_net = catd.WordNet()

doc_lens = [len(doc) for doc in word_net.docs]

# Plot
plt.figure(figsize=(16,7), dpi=160)
plt.hist(doc_lens, bins=100, color='navy')
# plt.text(750, 100, "Mean   : " + str(round(np.mean(doc_lens))))
# plt.text(750,  90, "Median : " + str(round(np.median(doc_lens))))
# plt.text(750,  80, "Stdev   : " + str(round(np.std(doc_lens))))
# plt.text(750,  70, "20%ile    : " + str(round(np.quantile(doc_lens, q=0.2))))
# # plt.text(750,  60, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))
#
# plt.gca().set(xlim=(0, 1000), ylabel='Number of Documents', xlabel='Document Word Count')
# plt.tick_params(size=16)
# plt.xticks(np.linspace(0, 100, 10))
# plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
# plt.show()
