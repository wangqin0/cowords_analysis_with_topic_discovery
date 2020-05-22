import catd
import os
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statistics

import heapq


if __name__ == '__main__':
    rows = catd.util.get_sql_database_input('weibo_COVID19.db')
    post_len = [len(row[0]) for row in rows]
    print(statistics.mean(post_len))
    print(statistics.median(post_len))