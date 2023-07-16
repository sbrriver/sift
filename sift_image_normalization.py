import os
from collections import defaultdict
from astropy.io import fits
import numpy as np
from PIL import Image

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

def normalize(img_dat):
    """Normalize image data to be black and white.

    Args:
        img_dat (ndarray): pixel data from fits file.

    Returns:
        Image: normalized image (type from PIL module).
    """
    maxval = np.max(img_dat)
    minval = np.min(img_dat)
    result = data_scale(img_dat, minval, maxval)
    
    #convert to jpg
    """scaling based on white level of image"""
    # """Subtract a number based on the average brightness of the image to remove faint brightness across image. Scale values above this to be brighter."""
    # zero_value_limit = np.mean(result.mean(axis=1)) * 240
    # scale_factor = 3000
    # image_data_jpg = (255*result - zero_value_limit) * scale_factor
    # """Clip the data to be within 0 to 255 before changing to 8 bit so values above 255 don't flip to lower values."""
    # image_data_jpg = np.clip(image_data_jpg, 0, 255).astype(np.uint8)
    # """If the average brightness is over threshold increase the zero_value_limit and rerun code. Lower threshold means this runs more and will take longer."""
    # threshold = 80
    # while np.mean(image_data_jpg.mean(axis=1)) > threshold:
    #     zero_value_limit += 0.02
    #     image_data_jpg = (255*result - zero_value_limit) * scale_factor
    #     image_data_jpg = np.clip(image_data_jpg, 0, 255).astype(np.uint8)
    # image = Image.fromarray(image_data_jpg, 'L')
    # return image
    
    """other methods"""
    """log"""
    # scale_factor = 5000
    # image_data_jpg = np.log(scale_factor * result + 1) / np.log(scale_factor) * 255
    # print(image_data_jpg)
    # image_data_jpg = np.clip(image_data_jpg, 0, 255).astype(np.uint8)
    # image = Image.fromarray(image_data_jpg, 'L')
    # return image
    
    """pow"""
    # scale_factor = 1000
    # image_data_jpg = (scale_factor ** result - 1) * 255
    # print(image_data_jpg)
    # image_data_jpg = np.clip(image_data_jpg, 0, 255).astype(np.uint8)
    # image = Image.fromarray(image_data_jpg, 'L')
    # return image
    
    """sqrt"""
    # image_data_jpg = np.sqrt(result) * 500
    # print(image_data_jpg)
    # image_data_jpg = np.clip(image_data_jpg, 0, 255).astype(np.uint8)
    # image = Image.fromarray(image_data_jpg, 'L')
    # return image
    
    """square""" 
    # image_data_jpg = result ** 2 * 1000000
    # print(image_data_jpg)
    # image_data_jpg = np.clip(image_data_jpg, 0, 255).astype(np.uint8)
    # image = Image.fromarray(image_data_jpg, 'L')
    # return image
    
    """asinh"""
    # scale_factor = 1000
    # image_data_jpg = scale_factor * np.sinh(result * 10) / 3
    # print(image_data_jpg)
    # image_data_jpg = np.clip(image_data_jpg, 0, 255).astype(np.uint8)
    # image = Image.fromarray(image_data_jpg, 'L')
    # return image

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

def data_process(directory):
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

        processed_unsorted_data.append([location, date, normalize(image_data), image_name, filter])

    for image in processed_unsorted_data:
        location = image[0]
        image_with_info = tuple(image[1:])
        sorted_data[location].append(image_with_info)

    return dict(sorted_data)

def data_store(normalized_images):
    """Go through normalized images and save in subfolder labelled by location.

    Args:
        normalized_images (defaultdict): dictionary of image data, with key being location (ra, dec)
        and value being a tuple of (date (string), normalized image (Image type), image name)
    """
    if not os.path.exists("training"):
        os.makedirs("training")

    for location in normalized_images.keys():
        location_path = "training/" + 'ra_'+location[0].replace(':','-').replace('+','p') + 'dec_'+location[1].replace(':','-').replace('+','p')
        if not os.path.exists(location_path):
            os.makedirs(location_path)
        for observation in normalized_images[location]:
            filter_path = location_path + "/" + observation[3]
            if not os.path.exists(filter_path):
                os.makedirs(filter_path)
            image = observation[1]
            image_name = observation[2]
            image.save(filter_path + "/" + image_name)

if __name__ == '__main__':
    processed_data = data_process(os.getcwd())
    data_store(processed_data)