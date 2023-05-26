import os
import re
from datetime import datetime, timedelta
import numpy as np
import ztfapi as z
import sift_image_normalization as norm

def date_range_generate(date):
    """Takes in date supernova occured and generates range of dates to get data from.
    Dates are 30 days on either end of the date supernova occurred.

    Args:
        date (str): date supernova occurred in YYYY-MM-DD format.

    Returns:
        tuple: (start, end), where start and end are strings in YYYY-MM-DD format.
    """
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    start = (date_obj - timedelta(days=30)).strftime('%Y-%m-%d')
    end = (date_obj + timedelta(days=30)).strftime('%Y-%m-%d')

    return start, end

#load list of supernovae
#max_rows: all the supernovae that have a date pre-2021
#usecols gets the ra, dec, date
supernovae_ra, supernovae_dec, supernovae_date = np.genfromtxt("tns-supernovae-list.csv", delimiter=",", skip_header=1, \
    usecols=(3,4,17), max_rows=7146, unpack=True, dtype=str, filling_values=0, encoding='mac-roman', invalid_raise=False)
#remove time from supernovae date so ztf api can use in search
supernovae_date = [s.split(' ')[0] for s in supernovae_date]

supernovae_list = list(zip(supernovae_ra, supernovae_dec, supernovae_date))
#create usable supernovae_list that removes any sub-list without a valid date element
pattern = r'^\d{4}-\d{2}-\d{2}$'
usable_supernovae_list = [supernova for supernova in supernovae_list if bool(re.match(pattern, supernova[2]))]

#download data
public_data_end = datetime(2021, 1, 20)
public_data_start = datetime(2018, 3, 20)
for supernova in usable_supernovae_list:
    supernova_date = datetime.strptime(supernova[2], '%Y-%m-%d')
    if supernova_date < public_data_end and supernova_date > public_data_start:
        start_date, end_date = date_range_generate(supernova[2])
        z.get_ztf_data(float(supernova[0]), float(supernova[1]), 0.01, start_date, end_date)

#process data into training data - normalization function needs adjustment
norm.data_process(os.getcwd())