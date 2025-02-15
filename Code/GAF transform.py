import os
import hdf5storage
import numpy as np
from tqdm import tqdm


# Normalize the data to [0, 1]
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Map normalized data to [-1, 1]
def scale_data(normalized_data):
    return 2 * normalized_data - 1


# Calculate the Gramian Angular Field matrices
def calculate_gaf_matrices(scaled_data):
    GASF = np.outer(scaled_data, scaled_data) - np.outer(np.sqrt(1 - scaled_data ** 2), np.sqrt(1 - scaled_data ** 2))
    GADF = np.outer(np.sqrt(1 - scaled_data ** 2), scaled_data) + np.outer(scaled_data, np.sqrt(1 - scaled_data ** 2))
    return GASF, GADF


data_path = 'Data/Seismic single track record/Class_1/'
datasets = os.listdir(data_path)

idx = 0
for tidx, lt in enumerate(datasets):
    mat = hdf5storage.loadmat(data_path + '/' + lt)
    records = mat['records']
    for i in tqdm(range(records.shape[1])):
        time_series = records[:, i]

        normalized_data = normalize_data(time_series)
        scaled_data = scale_data(normalized_data)

        GASF, GADF = calculate_gaf_matrices(scaled_data)

        summation_output_dir = 'Data/FIg-train-summation/class_1'
        difference_output_dir = 'Data/Fig-train-difference/class_1'

        gasf_filename = os.path.join(summation_output_dir, f'{idx}.npy')
        gadf_filename = os.path.join(difference_output_dir, f'{idx}.npy')
        idx += 1
        np.save(gasf_filename, GASF)
        np.save(gadf_filename, GADF)
