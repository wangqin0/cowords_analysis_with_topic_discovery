import os
import random

filename = 'tianya_posts_complete.txt'

with open(os.path.join('data', 'original_data', filename), 'r', ) as original_dataset:
    with open(os.path.join('data', 'original_data', 'sampling_' + filename), 'w+', ) as sampling_dataset:
        for line in original_dataset:
            print(line)
            if random.random() < 0.084:
                sampling_dataset.write(line)
