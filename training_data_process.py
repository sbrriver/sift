import os
import numpy as np
import ztfapi as z
import sift_image_normalization as norm

#load list of supernovae
#max_rows: all the supernovae that have a date pre-2021
#usecols gets the ra, dec, date
supernovae_ra, supernovae_dec, supernovae_date = np.genfromtxt("tns-supernovae-list.csv", delimiter=",", skip_header=1, \
    usecols=(3,4,17), max_rows=7146, unpack=True, dtype=str, filling_values=0, encoding='mac-roman', invalid_raise=False)
#remove time from supernovae date so ztf api can use in search
supernovae_date = [s.split(' ')[0] for s in supernovae_date]

supernovae_list = list(zip(supernovae_ra, supernovae_dec, supernovae_date))
#create usable supernovae_list that removes any sub-list with an empty string as an element
usable_supernovae_list = [supernova for supernova in supernovae_list if all(supernova)]
print(usable_supernovae_list)

#download data
for supernova in supernovae_list:
    z.get_ztf_data(float(supernova[0]), float(supernova[1]), 0.01, supernova[2])

#process data into training data - normalization function needs adjustment
#norm.data_process(os.getcwd())