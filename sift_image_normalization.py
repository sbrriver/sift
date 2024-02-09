import os
from collections import defaultdict
from astropy.io import fits
import numpy as np
from PIL import Image
import time
from scipy.ndimage import rotate

def data_scale(img_dat, minval, maxval):
    """Scale image data to [0, 1] range.

    Args:
        img_dat (ndarray): pixel data from fits file.
        minval (float): minimum pixel value.
        maxval (float): maximum pixel value.

    Returns:
        ndarray: rescaled pixel data.
    """
    #scale data to [0,1] range
    return (img_dat - minval)/(maxval - minval)

def normalize(img_dat, center_ra, center_dec, sn_ra, sn_dec, orientation, pixscale):
    """Normalize image data to be black and white.

    Args:
        img_dat (ndarray): pixel data from fits file.

    Returns:
        Image: normalized image (type from PIL module).
    """
    if center_ra != None:
        """Input supernovae ra and dec fits files are downloaded for."""
        """Will account for any tilt of image in a bit."""
        
        """Account for tilt of image"""
        tilt = orientation - 180
        img_dat = rotate(img_dat, tilt)
        """Should center exactly around sn, center = sn_ra, sn_dec"""
        center = (1536 - round((sn_ra - center_ra) * 3600/pixscale), 1540 - round((sn_dec - center_dec) * 3600/pixscale))
        pixel_location_ra = (sn_ra - center_ra) * 3600/pixscale
        zoom = 64
        img_dat = img_dat[0 if center[1] - zoom < 0 else center[1] - zoom: center[1] + zoom, 0 if center[0] - zoom < 0 else center[0] - zoom: center[0] + zoom] #(y1,y2:x1,x2)
        if (len(img_dat), len(img_dat[0])) != (zoom*2, zoom*2):
            return False
        maxval = np.max(img_dat)
        minval = np.min(img_dat)
        """Image Normalization"""
        result = data_scale(img_dat, minval, maxval)
        q = 12
        qth_percentile = np.percentile(result, [q])
        image_data_jpg = (result - qth_percentile) * 2000
        image_data_jpg = np.clip(image_data_jpg, 0, 255).astype(np.uint8)
        image = Image.fromarray(image_data_jpg, 'L')
        """Without Image Normalization"""
        result = data_scale(img_dat, minval, maxval) * 255
        # image_data_jpg = np.clip(result, 0, 255).astype(np.uint8)
        # image = Image.fromarray(image_data_jpg, 'L')
        
        return image
    
    else:
    #convert to jpg
        maxval = np.max(img_dat)
        minval = np.min(img_dat)
        result = data_scale(img_dat, minval, maxval)
        """percentile"""
        q = 45
        qth_percentile = np.percentile(result, [q])
        image_data_jpg = (result - qth_percentile) * 500000
        image_data_jpg = np.clip(image_data_jpg, 0, 255).astype(np.uint8)
        image = Image.fromarray(image_data_jpg, 'L')
        return image

def find_fits_files(directory):
    """Find all .fits files in given directory.

    Args:
        directory (string): path to directory to search. Generally pass in current working directory.

    Returns:
        list: list of paths to fits files.
    """
    fits_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".fits"):
                fits_files.append(os.path.join(root, file))

    return fits_files

def data_process(directory, center_ra, center_dec, sn_ra, sn_dec, orientation, pixscale):
    """Produces black and white jpgs from all fits files in given directory.

    Args:
        directory (string): path to directory to search for data.
    """
    processed_unsorted_data = []
    sorted_data = defaultdict(list)
    file_list = find_fits_files(directory)
    #iterate through files and normalize
    for file in file_list:
        header = fits.getheader(file)
        try:
            try:
                image_data = fits.getdata(file)
            except TypeError:#handles when file not entirely downloaded, etc
                continue
            date = header['OBSMJD']
            location = (header['RA'], header['DEC'])
            filter = header['FILTER']
        except KeyError:#in here to handle cases when file doesn't have date/ra/dec in header
            continue

        image_name = str(date) + os.path.basename(file).replace('.fits', '.jpg')

        processed_unsorted_data.append([location, date, normalize(image_data, center_ra, center_dec, sn_ra, sn_dec, orientation, pixscale), image_name, filter])

    for image in processed_unsorted_data:
        location = image[0]
        image_with_info = tuple(image[1:])
        sorted_data[location].append(image_with_info)

    return dict(sorted_data)

def data_store(normalized_images, center_ra, center_dec, sn_ra, sn_dec, version):
    """Go through normalized images and save in subfolder labelled by location.

    Args:
        normalized_images (defaultdict): dictionary of image data, with key being location (ra, dec)
        and value being a tuple of (date (string), normalized image (Image type), image name)
    """        
    if not os.path.exists("training"):
        os.makedirs("training")

    for location in normalized_images.keys():
        # location_path = "training/" + 'ra_'+location[0].replace(':','-').replace('+','p') + 'dec_'+location[1].replace(':','-').replace('+','p')
        # if not os.path.exists(location_path):
        #     os.makedirs(location_path)
        # for observation in normalized_images[location]:
        #     filter_path = location_path + "/" + observation[3]
        #     if not os.path.exists(filter_path):
        #         os.makedirs(filter_path)
        #     image = observation[1]
        #     image_name = observation[2]
        #     image.save(filter_path + "/" + image_name)
        """Changed path so each file is downloaded to the supernova it is meant for and to include ra and dec in degrees"""
        location_path = "training/" + str(sn_ra) + ", " + str(sn_dec)
        if not os.path.exists(location_path):
            os.makedirs(location_path)
        for observation in normalized_images[location]:
            image = observation[1]
            image_name = observation[2]
            image.save(location_path + "/" + version + '_' + str(center_ra) + ", " + str(center_dec) + "_" + image_name)

if __name__ == '__main__':
    start_time = time.time()
    processed_data = data_process(os.getcwd(), None, None, None, None)
    data_store(processed_data, None, None, None, None)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time to execute statement: " + str(elapsed_time) + 'seconds')
