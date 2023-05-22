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
    image_data_jpg = (255*result).astype(np.uint8)[::-1,:]
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
            image_data = fits.getdata(file)
            try:
                date = header['DATE']
                location = (header['PLATERA'], header['PLATEDEC'])
            except KeyError:
                continue
        except TypeError:
            continue

        image_name = os.path.basename(file).replace('.fits', '.jpg')#add date to name

        processed_unsorted_data.append([location, date, normalize(image_data), image_name])

    if not os.path.exists("training"):
        os.makedirs("training")

    for image in processed_unsorted_data:
        location = image[0]
        image_with_info = tuple(image[1:])
        sorted_data[location].append(image_with_info)

    return dict(sorted_data)

def data_store(normalized_images):
    """_summary_

    Args:
        normalized_images (defaultdict): dictionary of image data, with key being location (ra, dec)
        and value being a tuple of (date (string), normalized image (Image type), image name)
    """
    #go through normalized image keys and save in subfolder labelled by location. WRITE
    #can just do ImageType.save("training/"+image_name), just add in subfolder bit, etc
    return

if __name__ == '__main__':
    processed_data = data_process(os.getcwd())
    data_store(processed_data)
