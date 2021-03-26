import glob
import os


def get_labels():
    path = '../data/annotations'
    labels = []
    label_files = glob.glob(os.path.join(path, '*.txt'))
    label_files.sort()

    for index in range(0, 20000):
        labels.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    for label_index, file_path in enumerate(label_files):
        with open(os.path.join(os.getcwd(), file_path), 'r') as f:
            label = file_path.split('.')[0].split('/')[-1]

            for row in f:
                image_index = int(row.strip()) - 1
                labels[image_index][label_index] = 1

    return labels
