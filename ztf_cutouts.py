import re
import pandas as pd
from datetime import datetime, timedelta
from ztfquery import query
import multiprocessing
import requests
from concurrent.futures import ThreadPoolExecutor
import os
from astropy.time import Time
from astropy.io import fits
import numpy as np
from PIL import Image
from collections import defaultdict
import time
import argparse
import random

num_cores = multiprocessing.cpu_count()

save_path = 'cutouts'
zquery = query.ZTFQuery()

def load_supernovae_data():
    """Load supernovae data and preprocess it to get a usable list of supernovae."""
    supernovae_table = pd.read_csv("only sn - tns_public_objects.csv", delimiter=',', usecols=('ra', 'declination', 'time_received', 'source_group', 'filter'), dtype=str, encoding='mac-roman', on_bad_lines='warn')
    supernovae_table = supernovae_table[supernovae_table['source_group'] == 'ZTF']
    supernovae_table = supernovae_table[supernovae_table['filter'] == 'r']
    supernovae_table = supernovae_table[30:] #Filter out until 3 months have gone by
    supernovae_ra = supernovae_table['ra'].to_numpy()
    supernovae_dec = supernovae_table['declination'].to_numpy()
    supernovae_date = supernovae_table['time_received'].to_numpy()
    supernovae_group = supernovae_table['source_group'].to_numpy()
    supernovae_date = [s.split(' ')[0] for s in supernovae_date]
    supernovae_list = list(zip(supernovae_ra, supernovae_dec, supernovae_date, supernovae_group))
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    usable_supernovae_list = [supernova for supernova in supernovae_list if bool(re.match(pattern, supernova[2])) and supernova[3] == "ZTF"]
    return usable_supernovae_list

def query_and_generate_url(supernova):
    """Query for ZTF images based on supernova data and generate download URLs."""
    ra, dec, date_str, _ = supernova
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    group_global = os.environ.get('group_global', 'default_value')
    if group_global == 'sn':
        before_dates = ((date_obj - timedelta(days=70)).strftime('%Y-%m-%d'), (date_obj - timedelta(days=50)).strftime('%Y-%m-%d'))
        after_dates = ((date_obj - timedelta(days=1)).strftime('%Y-%m-%d'), (date_obj + timedelta(days=1)).strftime('%Y-%m-%d'))
    elif group_global == 'non':
        before_dates = ((date_obj + timedelta(days=70)).strftime('%Y-%m-%d'), (date_obj + timedelta(days=90)).strftime('%Y-%m-%d'))
        after_dates = ((date_obj + timedelta(days=130)).strftime('%Y-%m-%d'), (date_obj + timedelta(days=150)).strftime('%Y-%m-%d'))   
    elif group_global == 'star':
        before_dates = ((date_obj - timedelta(days=90)).strftime('%Y-%m-%d'), (date_obj - timedelta(days=50)).strftime('%Y-%m-%d'))
        after_dates = ((date_obj - timedelta(days=20)).strftime('%Y-%m-%d'), (date_obj + timedelta(days=20)).strftime('%Y-%m-%d'))
    
    image_size = 128.55  # Define the size of the cutout in arcseconds
    
    # Define time ranges for before and during the supernova event
    time_ranges = [before_dates, after_dates]
    urls = []
    fail = 0
    #creates url for before and during sn
    if group_global == 'star':
        ra, dec = random_ra_dec()
        
    for time_range in time_ranges:
        start_mjd = Time(time_range[0], format='iso').jd
        end_mjd = Time(time_range[1], format='iso').jd
        search = f"seeing<2 and filtercode='zr' and obsjd BETWEEN {start_mjd} and {end_mjd}"
        zquery.load_metadata(kind='sci', radec=[ra, dec], mcen=True, sql_query=search)
        image = zquery.metatable
        if not image.empty:
            urls.append(generate_url(image, ra, dec, image_size))  # Generate URL for the first image in the time range
        else:
            fail = 1
    if not fail:
        return urls

def random_ra_dec():
    ra = random.uniform(0, 360)  # RA ranges from 0 to 360 degrees
    dec = random.uniform(0, 90)  # Dec ranges from 0 to +90 degrees for the Northern sky
    return ra, dec

def generate_url(image, ra, dec, image_size):
    """Generate the ZTF cutout URL based on image metadata."""
    # added image_size parameter
    """Create a function for generating ZTF cutout URLs:
    - Takes in the full-sized image, and the ra and dec of the object
    - Gets all relevant information from the full image
    - Returns the cutout URL for that object"""
    year = image['obsdate'].values[0][0:4]
    month = image['obsdate'].values[0][5:7]
    day = image['obsdate'].values[0][8:10]
    filefracday = str(image['filefracday'].values[0])
    fracday = filefracday[8:14]
    imgtypecode = str(image['imgtypecode'].values[0])
    qid = str(image['qid'].values[0])
    
    # Get the ZTF field and pad it to 6 digits
    field = str(image['field'].values[0])
    if len(field) < 6:
        pad_field = 6 - len(field)
        field = '0'*pad_field+str(field)
    
    filtercode = image['filtercode'].values[0]
    
    # Get the CCD ID and pad it to 2 digits
    ccdid = str(image['ccdid'].values[0])
    if len(ccdid) < 2:
        pad_ccdid = 2 - len(ccdid)
        ccdid = '0'*pad_ccdid+str(ccdid)

    cut_out = 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/'+year+'/'+month+day+'/'+fracday+'/ztf_'+filefracday+'_'+field+'_'+filtercode+'_c'+ccdid+'_'+imgtypecode+'_q'+qid+'_sciimg.fits?center='+str(ra)+','+str(dec)+'&size='+str(image_size)+'arcsec&gzip=false'
    
    return cut_out

def download_image(url, before_or_after, number):
    """Download an image from a given URL."""
    group_global = os.environ.get('group_global', 'default_value')
    
    if not url:
        return
    # Extracting the filename from the URL to keep track of before/after pairs
    split = url.split('/')[-1]
    filename = split[:49]
    center = split[50:][:-26]
    
    directory = os.path.join(save_path, group_global)
    # Define your directory where you want to save images
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for existing_file in os.listdir(directory):
        if existing_file.startswith(str(number)):
            print(f"File starting with {number} already exists. Skipping download.")
            return  # Skip downloading this file
    
    full_path = os.path.join(directory, f"{number}_{before_or_after}_{center}_{filename}")
        
    # Download and save the image
    response = requests.get(url)
    if response.status_code == 200:
        with open(full_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {url}")

def download_images(urls, lower_index, upper_index):
    """Download images from a list of URLs using ThreadPoolExecutor for concurrency."""
    # Using ThreadPoolExecutor to manage a pool of threads for downloading images
    length = len(urls)
    before_or_afters = [i % 2 for i in range(length)]
    numbers = [i // 2 + 1 for i in range(length)]
    # numbers = [i // 2 + 1 for i in range(lower_index, upper_index)][:length*2]
    with ThreadPoolExecutor(max_workers=num_cores*3) as executor:  # Adjust max_workers as needed
        executor.map(download_image, urls, before_or_afters, numbers)
        
def process_fits(fits_path):
    """Open a FITS file, normalize its data, and return a PIL image."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        # Normalize the image data
        data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
        # Convert data to a PIL image (grayscale) after scaling to 0-255 range
        image = Image.fromarray((data * 255).astype(np.uint8), 'L')
    return image, data.shape

def crop_to_128x128(image, original_shape):
    """Crop the given image to 128x128 if it's larger, or return None if smaller."""
    if original_shape[0] < 128 or original_shape[1] < 128:
        print("Image is smaller than 128x128. Operation aborted.")
        return None
    else:
        # Calculate the cropping box
        left = (original_shape[1] - 128) // 2
        upper = (original_shape[0] - 128) // 2
        right = left + 128
        lower = upper + 128
        return image.crop((left, upper, right, lower))

def concatenate_and_save_images(before_fits, during_fits, output_path):
    """Concatenate 'before' and 'during' images side by side and save as JPEG."""
    img_before, shape_before = process_fits(before_fits)
    img_during, shape_during = process_fits(during_fits)
    
    img_before = crop_to_128x128(img_before, shape_before)
    img_during = crop_to_128x128(img_during, shape_during)
    
    if img_before is None or img_during is None:
        return  # Aborts if either image was smaller than 128x128 and couldn't be cropped
    
    # Create a new image with a width that is the sum of the two images and the height of the tallest image
    dst = Image.new('L', (img_before.width + img_during.width, max(img_before.height, img_during.height)))
    dst.paste(img_before, (0, 0))
    dst.paste(img_during, (img_before.width, 0))
    
    # Save the concatenated image
    dst.save(output_path)
    
def process_downloaded_images(directory):
    """Process all downloaded FITS pairs, concatenate them, and save as JPEGs."""
    fits_files = sorted([f for f in os.listdir(directory) if f.endswith('.fits')])
    
    # Group files by supernova identifier
    supernovae = defaultdict(lambda: {'before': None, 'during': None})
    for fits_file in fits_files:
        parts = fits_file.split('_')
        sn_id, time_id, center = parts[0], parts[1], parts[2]  # e.g., '50' as sn_id and '0' as time_id for before
        if time_id == '0':
            supernovae[sn_id]['before'] = fits_file
        elif time_id == '1':
            supernovae[sn_id]['during'] = fits_file
    # Iterate over grouped files and concatenate 'before' and 'during' images
    images_directory = os.path.join(directory, 'images')
    if not os.path.exists(images_directory):
                os.makedirs(images_directory)
    for sn_id, files in supernovae.items():
        already_downloaded = any(file.startswith(sn_id) for file in os.listdir(images_directory))
        if already_downloaded:
            print(f"Processed image for SN {sn_id} already exists. Skipping.")
            continue 
            
        if files['before'] and files['during']:
            before_fits_path = os.path.join(directory, files['before'])
            during_fits_path = os.path.join(directory, files['during'])
            output_path = os.path.join(images_directory, f"{sn_id}_{center}_concatenated.jpg")
            concatenate_and_save_images(before_fits_path, during_fits_path, output_path)

def delete_fits_files(directory):
    """
    Deletes all FITS files in the specified directory.

    Parameters:
    - directory: The path to the directory from which FITS files will be deleted.
    """
    # List all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a FITS file
        if filename.endswith(".fits"):
            # Construct the full path to the file
            file_path = os.path.join(directory, filename)
            # Delete the file
            os.remove(file_path)
        
def main(group):
    os.environ['group_global'] = group
    start_time = time.time()
    usable_supernovae_list = load_supernovae_data()
    lower_index = 0
    upper_index = 0
    usable_supernovae_list = usable_supernovae_list[lower_index:] #2811 total
    # Utilize multiprocessing to handle queries for multiple supernovae
    pool = multiprocessing.Pool(processes=num_cores)
    urls_list = pool.map(query_and_generate_url, usable_supernovae_list)
    # Go from list of lists to one big list
    urls_list = [urls for urls in urls_list if urls is not None]
    urls = [url for sublist in urls_list for url in sublist]  # Flatten the list of lists
    download_images(urls, lower_index, upper_index)
    group_global = os.environ.get('group_global', 'default_value')
    directory = os.path.join(save_path, group_global)
    process_downloaded_images(directory)
    delete_fits_files(directory)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time to load and download: " + str(elapsed_time) + ' seconds')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
        'Download cutouts for training data.')
    parser.add_argument('--group', type=str, required=True, \
        help="'sn' for galaxies with supernovae, 'non' for galaxies without supernovae, and 'star' for non-galaxy images")
    args = parser.parse_args()
    # Call the function to retrieve the objects
    try:
        main(args.group)
    except Exception as e:
        print(f'Error retrieving objects: {e}')