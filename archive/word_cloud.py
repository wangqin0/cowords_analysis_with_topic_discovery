# importing all necessery modules
from wordcloud import WordCloud, STOPWORDS
import catd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')

word_net = catd.util.load_obj('reduced_weibo_COVID19_complete')

corpus = word_net.get_cut_corpus()

# Join the different processed titles together.
long_string = ''
for doc in corpus:
    for word in doc:
        long_string += word + ', '

wordcloud = WordCloud(width=1920, height=1920,
                      background_color='white',
                      font_path='../data/STHeiti_Medium.ttc',
                      min_font_size=10, max_font_size=400,
                      collocations=False).generate(long_string)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
