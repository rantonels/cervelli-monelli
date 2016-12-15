## Downsample brain scan data

import numpy as np
import nibabel as nib
import os
import scipy.signal
from tqdm import tqdm
import pickle
import csv

t_side = 10
n_split = 10

target_shape = np.array([t_side,t_side,t_side,1])

fisher = np.load("data/fisher.npz")["arr_0"]
quantile_threshold = 0.95
fisher_threshold = scipy.stats.mstats.mquantiles(
    fisher[fisher > 0].ravel(),
    prob=quantile_threshold)[0]

fisher_mask = fisher > fisher_threshold


def open_and_flatten(filename):

    brain_scan = nib.load(filename)
    brain_array = brain_scan.get_data().mean(-1)
#
#
    mean = np.mean(brain_array)
    median = np.median(brain_array)
    variance = np.var(brain_array)
#    grey_matter = np.sum(np.greater(brain_array,mean*0.5))/float(np.size(brain_array))
#
#
    brain_trunc = brain_array# brain_array[60:150,40:150,30:150]
# 

    csf = np.sum(np.logical_and(brain_array > 0, brain_array < 300))
    gm = np.sum(np.logical_and(brain_array > 650, brain_array < 850))
    wm = np.sum(np.logical_and(brain_array > 1250, brain_array < 1450))

    #print csf,gm,wm

    #FISHER
    
    fisher_feature = np.einsum("ijk,ijk",brain_trunc, fisher_mask)
    fisher_features = brain_trunc[ fisher_mask ]

    fourier_brain = np.fft.fftn(brain_trunc)


    fourier_slice = fourier_brain[0:3,0:3,0:3]

    re = np.real(fourier_slice)
    re = re.reshape((re.size,))
    im = np.imag(fourier_slice)[1:]
    im = im.reshape((im.size,))

    out_fourier = np.concatenate((re,im)) * 1e-8
#
# 
#    # histograms
#
#    H = 8
#
##    chunks_shape = (np.array(brain_trunc.shape) / np.array([t_side,t_side,t_side]) ).astype(int)
#
#
#    def spl(arr,ax):
#        return np.array(np.split(arr,n_split,axis=ax))
#
#    chunks = spl(spl(spl(brain_trunc,-1),-2),-3)
#
#    histograms = np.zeros( (chunks.shape[0:3] )+( H,)  )
#    histograms += 3
#
#    #for x in range(chunks_shape[0]):
#    #    for y in range(chunks_shape[1]):
#    #        for z in range(chunks_shape[2]):
#    #            chunk = brain_trunc[    x*t_side : (x+1)*t_side ,
#    #                                    y*t_side : (y+1)*t_side,
#    #                                    z*t_side : (z+1)*t_side
#    #                                    ]
#    for x in range(n_split):
#     for y in range(n_split):
#      for z in range(n_split):
#        chunk = chunks[z,y,z,:,:,:]
#        chunk = chunk.reshape((chunk.size,))
#        histo, bin_edges =  np.histogram(chunk, H, range = (np.min(brain_trunc),np.max(brain_trunc) ) )
#
#        histograms[x,y,z,:] = histo
#
#    histograms = histograms.reshape((histograms.size,))
#
#


#    down_arr = scipy.signal.decimate(brain_trunc, brain_trunc.shape[0]//t_side, axis = 0)
#    down_arr = scipy.signal.decimate(down_arr, down_arr.shape[1]//t_side, axis = 1)
#    down_arr = scipy.signal.decimate(down_arr, down_arr.shape[2]//t_side, axis = 2)
#
#    down_arr = down_arr.reshape((down_arr.size,))

#
#    import matplotlib.pyplot as plt
#
#    for i in range(t_side):
#        plt.imshow(down_arr[:,:,i])
#        plt.show()


    return  np.concatenate((
        np.array([mean,median,variance,csf,gm,wm,fisher_feature]),
        np.array([]),
        fisher_features,
#        out_fourier
        ))
#        histograms
        
        


    return np.array([mean])


    
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

train_dataset = []

def load_features_abis(fname):
    out = {}
    with open(fname,'rb') as csvfile:
        reader = csv.reader(csvfile,delimiter = ",")
        first_row = True
        for row in reader:
            if first_row:
                first_row = False
                continue
            fid,data = int(row[0]), row[4:]
            out[fid] = np.array(data)
    return out

features_abis = load_features_abis("features/features_abis.csv")
features_abis_test = load_features_abis("features/features_abis-test.csv")


for index in tqdm(range(1,entries_total+1)):
    flattened = open_and_flatten("data/cropped/train_%d_cropped.nii.gz"%index)
    flattened_abis = np.array(features_abis[index])
    flattened = np.concatenate((flattened,flattened_abis))
    train_dataset.append(flattened)

print("Converting to np array...")

train_dataset = np.array(train_dataset)

print("Saving...")
np.save(os.path.join(out_path,"train_dataset"),train_dataset)


test_dataset = []

for index in tqdm(range(1,entries_test+1)):
    flattened = open_and_flatten("data/cropped/test_%d_cropped.nii.gz"%index)
    flattened_abis = np.array(features_abis_test[index])
    flattened = np.concatenate((flattened,flattened_abis))
    test_dataset.append(flattened)


print("Converting to np array...")

test_dataset = np.array(test_dataset)

print("Saving...")
np.save(os.path.join(out_path,"test_dataset"),test_dataset)







