import catd
import os
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statistics

import heapq


if __name__ == '__main__':
    rows = catd.util.vis_post_num_time_stats('weibo_COVID19.db')
