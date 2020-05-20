import os
import catd

dir = os.path.join('output', 'objects')
for word_net_object in os.listdir(dir):
    if '.pkl' in word_net_object and 'weibo_COVID19_reduced_' in word_net_object:
        object_name = word_net_object.split('.pkl')[0]
        word_net = catd.util.load_obj(object_name)
        print(object_name)
        print(word_net.description())
