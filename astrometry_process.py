import os
from astroquery.astrometry_net import AstrometryNet
from astropy.table import Table
import requests
import json
from astropy.config import reload_config
from astropy.io import fits
from astrometry.net.client import Client
import sift_image_normalization as sin
import time

def astrometry(directory):
    """Uploads fits files in directory to astrometry.net, runs sift_image_normalization.

    Args:
        directory (string): path to directory to search for data.
    """
    
    """Login, change api_key to your key found at https://nova.astrometry.net/api_help."""
    api_key = 'iqzsxmsaxydxbioo'
    client = Client()
    client.login(api_key)

    file_list = sin.find_fits_files(directory)
    count = 0
    for file in file_list:
        count += 1
        print('\n' + str(count) + '/' + str(len(file_list)) + '\n')
        upload = client.upload(fn=file)
        subid = str(upload['subid'])
        
        """Getting jobid. If the job_calibrations list is empty, calibration is not complete yet and will wait 10 seconds."""
        job_calibrations_length = 0
        while job_calibrations_length == 0:
            print('\n' + 'Request while loop running')
            status = requests.post('http://nova.astrometry.net/api/submissions/' + subid, headers={'Referer': 'http://www.example.com'})
            job_calibrations_length = len(status.json()['job_calibrations'])
            time.sleep(10)
        jobid = str(status.json()['jobs'])[1:-1]
        
        """Getting calibration data"""
        calibration = client.send_request('jobs/%s/calibration' % jobid)
        center_ra = calibration['ra']
        center_dec = calibration['dec']
        # width_arcsec = calibration['width_arcsec']
        # height_arcsec = calibration['height_arcsec']
        # radius = calibration['radius']
        # orientation = calibration['orientation']
        # annotations = client.send_request('jobs/%s/annotations' % jobid)
        # info = client.send_request('jobs/%s/info' % jobid)
        # client.annotate_data(jobid)
        # client.job_status(jobid)
        print(center_ra, center_dec)
        
        """Image normalization"""
        processed_data = sin.data_process(file[0:60], center_ra, center_dec)
        sin.data_store(processed_data)
    
if __name__ == '__main__':
    start_time = time.time()
    astrometry(os.getcwd())
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time to execute statement: " + str(elapsed_time) + 'seconds')