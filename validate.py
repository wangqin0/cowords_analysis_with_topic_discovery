from dumpObj import *

edges = load_obj('wordNet_edge')

total = 0
error_counter = 0
for word in edges:
    for nb_word in edges[word]:
        total += 1
        if nb_word == 'self_freq':
            continue
        if edges[word][nb_word] > edges[nb_word]['self_freq']:
            print('edge: ' + word + '_' + nb_word + ': ' + str(edges[word][nb_word]) + '\t' + str(edges[nb_word]['self_freq']))
            error_counter += 1
print('total: ' + str(total))
print('error counter: ' + str(error_counter))
print('error rate: ' + str(error_counter/total))