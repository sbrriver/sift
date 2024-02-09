import os
import re
from datetime import datetime, timedelta
import numpy as np
import ztfapi as z
import sift_image_normalization as norm
import astrometry_process as a
import pandas as pd
from astrometry.net.client import Client

def date_range_generate(date):
    """Takes in date supernova occured and generates range of dates to get data from.
    Dates are 30 days on either end of the date supernova occurred.

    Args:
        date (str): date supernova occurred in YYYY-MM-DD format.

    Returns:
        tuple: (start, end), where start and end are strings in YYYY-MM-DD format.
    """
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    start = ((date_obj - timedelta(days=90)).strftime('%Y-%m-%d'), (date_obj - timedelta(days=30)).strftime('%Y-%m-%d'))
    # end = (date_obj + timedelta(days=30)).strftime('%Y-%m-%d')
    """Currently going from 3 month before date to date"""
    end = ((date_obj - timedelta(days=1)).strftime('%Y-%m-%d'), (date_obj + timedelta(days=1)).strftime('%Y-%m-%d'))
    
    return start, end

#load list of supernovae
#max_rows: all the supernovae that have a date pre-2021
#usecols gets the ra, dec, date
supernovae_table = pd.read_csv("only sn - tns_public_objects.csv", delimiter=',', usecols=('ra', 'declination', 'time_received', 'source_group'), dtype=str, encoding='mac-roman', on_bad_lines='warn')
supernovae_ra = supernovae_table['ra'].to_numpy()
supernovae_dec = supernovae_table['declination'].to_numpy()
supernovae_date = supernovae_table['time_received'].to_numpy()
supernovae_group = supernovae_table['source_group'].to_numpy()
#remove time from supernovae date so ztf api can use in search
supernovae_date = [s.split(' ')[0] for s in supernovae_date]

supernovae_list = list(zip(supernovae_ra, supernovae_dec, supernovae_date, supernovae_group))
#create usable supernovae_list that removes any sub-list without a valid date element
pattern = r'^\d{4}-\d{2}-\d{2}$'
usable_supernovae_list = [supernova for supernova in supernovae_list if bool(re.match(pattern, supernova[2])) and supernova[3] == "ZTF"]
#download data
public_data_end = datetime(2019, 7, 30)
public_data_start = datetime(2018, 3, 15)

count = 0
api_key = 'iqzsxmsaxydxbioo'
client = Client()
client.login(api_key)
    
for supernova in usable_supernovae_list:
    supernova_date = datetime.strptime(supernova[2], '%Y-%m-%d')
    if supernova_date < public_data_end and supernova_date > public_data_start and count < 101:
        count += 1
        print(str(count) + " / " + str(len(usable_supernovae_list)) + " supernovae")
        start_date, end_date = date_range_generate(supernova[2])        
        file_paths = z.get_ztf_data(float(supernova[0]), float(supernova[1]), 0.1, start_date, end_date, True)
        print("SN data:" + str(supernova))
        if file_paths and not os.path.exists("training/" + (supernova[0]) + ", " + supernova[1]):
            a.astrometry(os.getcwd() + file_paths[0][:0:-1][:49:-1], float(supernova[0]), float(supernova[1]), client, 'before')
            a.astrometry(os.getcwd() + file_paths[1][:0:-1][:49:-1], float(supernova[0]), float(supernova[1]), client, 'after')

#process data into training data - normalization function needs adjustment
# norm.data_process(os.getcwd())
