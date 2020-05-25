import matplotlib

import catd
import os
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statistics

import heapq


if __name__ == '__main__':
    x = range(2, 21, 1)

    coherence_values = [0.4416, 0.4885, 0.515, 0.6087, 0.5897, 0.6706, 0.6523, 0.709, 0.643, 0.7072, 0.6875, 0.6793, 0.7033, 0.6976, 0.6798, 0.7061, 0.6909, 0.6654, 0.641]
    print(len(coherence_values))
    for index, cv in zip(x, coherence_values):
        print("Num Topics =", index, " has Coherence Value of", round(cv, 4))

    chinese_font = FontProperties(fname=os.path.join('data', 'STHeiti_Medium.ttc'))

    matplotlib.use('TkAgg')
    plt.plot(x, coherence_values)
    plt.xticks(x)
    plt.xlabel("主题数量", fontproperties=chinese_font)
    plt.ylabel("Coherence 数值", fontproperties=chinese_font)
    plt.show()