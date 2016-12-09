## Downsample brain scan data

import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import scipy.signal
from tqdm import tqdm
import pickle

t_side = 3

target_shape = np.array([t_side,t_side,t_side,1])

def open_and_flatten(filename):
    brain_scan = nib.load(filename)
    brain_array = brain_scan.get_data().mean(-1)
    
    ceil_shape = (t_side*np.ceil(np.array(brain_array.shape[0:3])/float(t_side))).astype(int)

    padded_array = np.zeros(ceil_shape,dtype = np.float)
    padded_array[:brain_array.shape[0],:brain_array.shape[1],:brain_array.shape[2]] = brain_array

    downsampled_array = scipy.signal.decimate(padded_array,
            brain_array.shape[0] // target_shape[0],
            axis = 0)
    downsampled_array = scipy.signal.decimate(downsampled_array,
            brain_array.shape[1] // target_shape[1],
            axis = 1)
    downsampled_array = scipy.signal.decimate(downsampled_array,
            brain_array.shape[2] // target_shape[2],
            axis = 2)

    return downsampled_array



entries_total = 278
entries_test = 138

out_path = "features/"

d = os.path.dirname(out_path)
if not os.path.exists(d):
    os.makedirs(d)

train_dataset = {}

for index in tqdm(range(1,entries_total+1)):
    flattened = open_and_flatten("set_train/train_%d.nii"%index)
    train_dataset[index] = flattened

pickle.dump(train_dataset, open(os.path.join(out_path,"train_dataset"),'w'))

test_dataset = {}

for index in tqdm(range(1,entries_test+1)):
    flattened = open_and_flatten("set_test/test_%d.nii"%index)
    test_dataset[index] = flattened


pickle.dump(test_dataset, open(os.path.join(out_path,"test_dataset"),'w'))





