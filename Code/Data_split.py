import os
import pickle as pkl
import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
import torch

summation_data_path = 'Data/Fig-train-summation/'
difference_data_path = 'Data/Fig-train-difference/'
datasets = os.listdir(summation_data_path)

images = np.zeros((1939, 2, 50, 50))
labels = np.zeros((1939, 5))

idx = 0
for tidx, lt in enumerate(datasets):
    summation_filenames = os.listdir(summation_data_path + lt)
    difference_filenames = os.listdir(difference_data_path + lt)

    summation_filenames.sort(key=lambda x: int(x[:-4]))
    difference_filenames.sort(key=lambda x: int(x[:-4]))

    for sum_fn, diff_fn in zip(summation_filenames, difference_filenames):
        summation_img = np.load(summation_data_path + lt + '/' + sum_fn)
        difference_img = np.load(difference_data_path + lt + '/' + diff_fn)

        images[idx, 0] = summation_img
        images[idx, 1] = difference_img
        labels[idx, tidx] = 1
        idx += 1

data, test_data, labels, test_labels = train_test_split(images, labels, test_size=0.1, random_state=42)
train_images, validation_images, train_labels, validation_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

train = data_utils.TensorDataset(torch.from_numpy(train_images).float(), torch.from_numpy(train_labels).float())
validation = data_utils.TensorDataset(torch.from_numpy(validation_images).float(), torch.from_numpy(validation_labels).float())
prediction = data_utils.TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).float())

with open('Datasets/train.pkl', 'wb') as f:
    pkl.dump(train, f)

with open('Datasets/validation.pkl', 'wb') as f:
    pkl.dump(validation, f)

with open('Datasets/prediction.pkl', 'wb') as f:
    pkl.dump(prediction, f)