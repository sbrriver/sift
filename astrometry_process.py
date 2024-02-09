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
import urllib.error

def astrometry(directory: str, sn_ra, sn_dec, client, version):
    """Uploads fits files in directory to astrometry.net, runs sift_image_normalization.

    Args:
        directory: path to directory to search for data.
    """
    
    """Login, change api_key to your key found at https://nova.astrometry.net/api_help."""
    print("\n Astrometry running on: " + str(directory) + "\n")
    # api_key = 'iqzsxmsaxydxbioo'
    # client = Client()
    # client.login(api_key)

    file = sin.find_fits_files(directory)[0]
    # count = 0
    # count += 1
    # print(str(count) + '/' + str(len(file_list)) + ' fits files')
    retries = 3
    for i in range(retries):
        try:
            upload = client.upload(fn=file)
            if type(upload) == dict:
                subid = str(upload['subid'])
            else:
                upload = client.upload(fn=file)
                time.sleep(1)
                try:
                    subid = str(upload['subid'])
                except:
                    return
            status = requests.post('http://nova.astrometry.net/api/submissions/' + subid, headers={'Referer': 'http://www.example.com', 'scale_type': 'ev', 'scale_est': '1.0123535066635947', 'scale_err': '1'})
            break
        except urllib.error.URLError as e:
            print(f"Attempt {i+1} failed: {e}")
            time.sleep(5)
    else:
        print("All retry attempts failed. Check your network or server.")
    time.sleep(10)
    job_calibrations_length = 0
    loops = 0
    """Getting jobid. If the job_calibrations list is empty, calibration is not complete yet and will wait."""
    while job_calibrations_length == 0 and loops < 40:
        loops += 1
        time.sleep(2)
        status = requests.post('http://nova.astrometry.net/api/submissions/' + subid, headers={'Referer': 'http://www.example.com', 'scale_type': 'ev', 'scale_est': '1.0123535066635947', 'scale_err': '1'})
        job_calibrations_length = len(status.json()['job_calibrations'])
    if loops == 40:
        print('\n\n\n image took too long to solve \n\n\n')
    else:
        jobid = str(status.json()['jobs'])[1:-1]
        """Getting calibration data"""
        calibration = client.send_request('jobs/%s/calibration' % jobid)
        center_ra = calibration['ra']
        center_dec = calibration['dec']
        # width_arcsec = calibration['width_arcsec']
        # height_arcsec = calibration['height_arcsec']
        # radius = calibration['radius']
        pixscale = calibration['pixscale']
        orientation = calibration['orientation']
        # annotations = client.send_request('jobs/%s/annotations' % jobid)
        # info = client.send_request('jobs/%s/info' % jobid)
        # client.annotate_data(jobid)
        # client.job_status(jobid)
        print("Image center: (" + str(center_ra) + ", " + str(center_dec) + ")")
        """Image normalization"""

        try:
            processed_data = sin.data_process(file[:-49], center_ra, center_dec, sn_ra, sn_dec, orientation, pixscale)
            sin.data_store(processed_data, center_ra, center_dec, sn_ra, sn_dec, version)
        except Exception as e:
            if e is KeyboardInterrupt:
                exit()
            print("\n\n\n An exception has occured! Image not downloaded \n\n\n")

if __name__ == '__main__':
    start_time = time.time()
    api_key = 'iqzsxmsaxydxbioo'
    client = Client()
    client.login(api_key)
    #Just using random sn for example here
    astrometry(os.getcwd(), 210.910674637, 54.3116510708, client)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time to execute statement: " + str(elapsed_time) + 'seconds')
