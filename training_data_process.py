import os
import numpy as np
import ztfapi as z
import sift_image_normalization as norm

#load list of supernovae - FIX
#max_rows: all the supernovae that have a date pre-2021
#usecols gets the ra, dec, date
supernovae_ra, supernovae_dec, supernovae_date = np.loadtxt("tns-supernovae-list.csv", delimiter=",", skiprows=1, \
    usecols=(3,4,17), max_rows=7146, unpack=True, dtype=str)
print(supernovae_date)#test line remove

supernovae_list = list(zip(supernovae_ra, supernovae_dec, supernovae_date))

#download data
for supernova in supernovae_list:
    z.get_ztf_data(supernova[0], supernova[1], 0.01, supernova[2])

#process data into training data - normalization function needs adjustment
norm.data_process(os.getcwd())