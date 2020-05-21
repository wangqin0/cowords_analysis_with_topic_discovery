import sqlite3
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pylab import mpl


def draw_post_num_time_stats(database_filename):
    con = sqlite3.connect(os.path.join('data', 'original_data', database_filename))
    cursor = con.cursor()
    cursor.execute("SELECT post_time, COUNT(1) AS post_num FROM posts GROUP BY post_time")
    rows = cursor.fetchall()

    # plt.gca().xaxis.set_major_formatter(matdates.DateFormatter('%m/%d/%Y'))
    # plt.gca().xaxis.set_major_locator(matdates.DayLocator())
    #
    # plt.plot(x, y)
    # plt.gcf().autofmt_xdate()
    # plt.show()

    x = [datetime.strptime(row[0], '%Y%m%d').date() for row in rows]
    y = [row[1] for row in rows]

    fig, ax = plt.subplots()

    # 配置横坐标
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.set(xlabel='日期', ylabel='收集微博数（条）',
           title='每日收集微博数量')
    ax.grid()

    # Plot
    ax.plot(x, y)
    fig.autofmt_xdate()  # 自动旋转日期标记

    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    plt.show()

    return rows


if __name__ == '__main__':
    dates = draw_post_num_time_stats('weibo_COVID19.db')



# import scipy.stats as st
# plt.hist(x, density=True, bins=30, label="Data")
# mn, mx = plt.xlim()
# plt.xlim(mn, mx)
# kde_xs = np.linspace(mn, mx, 301)
# kde = st.gaussian_kde(x)
# plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
# plt.legend(loc="upper left")
# plt.ylabel('Probability')
# plt.xlabel('Data')
# plt.title("Histogram");