import glob
import os
import pprint
import numpy as np

path = 'data/raw/annotations'

image_index_labels = {}

labels = []

for file_path in glob.glob(os.path.join(path, '*.txt')):
    with open(os.path.join(os.getcwd(), file_path), 'r') as f:
        label = file_path.split('.')[0].split('/')[-1]

        for row in f:
            image_index = row.strip()
            image_index_labels[image_index] = label


for index in range(1, 20001):
    try:
        labels.append(image_index_labels[str(index)])
    except:
        labels.append('other')

with open('data/labels.txt', 'w') as f:
    f.write('\n'.join(labels) + '\n')
