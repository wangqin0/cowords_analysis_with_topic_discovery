import catd

word_net = catd.util.load_obj('reduced_weibo_COVID19_complete')
word_net.export_for_gephi()
print('completed')