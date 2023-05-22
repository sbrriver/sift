import os
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

#find all .fits files
fits_files = []
for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        if file.endswith(".fits"):
            fits_files.append(os.path.join(root, file))

result_image_data = []

#iterate through files and normalize
for file in fits_files:
    #header = fits.getheader(file)
    try:
        image_data = fits.getdata(file)
    except TypeError:
        continue

    image_name = os.path.basename(file).replace('.fits', '.jpg')

    if not os.path.exists("training"):
        os.makedirs("training")
    normalize(image_data).save("training/"+image_name)
