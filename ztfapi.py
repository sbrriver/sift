import argparse
import os
from ztfquery import query
from astropy.time import Time
import lightkurve as lk
import pandas as pd

def get_ztf_data(ra, dec, radius, start_date, end_date, filter=None):
    """Gets specified data from the ztf database. Only can access public data.

    Args:
        ra (float): right ascension
        dec (float): declination
        radius (float): radius of region of sky to search, with center at (ra, dec), in arcsec
        start_date (str): starting date to start search for
        end_date (str): end date of time to search in
    """
    zquery = query.ZTFQuery()

    # convert start_date and end_date to MJD
    start_mjd = Time(start_date, format='iso').mjd
    end_mjd = Time(end_date, format='iso').mjd

    zquery.load_metadata(radec=[ra,dec], size=radius, sql_query=f"obsjd BETWEEN {start_mjd} AND {end_mjd}")
    zquery.download_data("sciimg.fits", show_progress=True, nprocess=1, verbose=True, \
        overwrite=False)

def get_tess_data(ra, dec, radius, start_date, end_date):
    """Gets specified data from the tess database. Only can access public data.

    Args:
        ra (float): right ascension
        dec (float): declination
        radius (float): unused
        start_date (str): unused
        end_date (str): unused
    """
    target = str(ra) + " " + str(dec)
    image_result = lk.search_tesscut(target)
    if image_result:
        image_result.download_all(download_dir=os.getcwd())
        print(f'{len(image_result)} images downloaded.')
    else:
        print('No objects found with given search parameters.')

databases = {'ztf':get_ztf_data, 'tess':get_tess_data}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=\
        'Retrieve ZTF objects within a certain area and time range.')
    parser.add_argument('--ra', type=float, required=True, \
        help='Right ascension of center of search area.')
    parser.add_argument('--dec', type=float, required=True, \
        help='Declination of center of search area.')
    parser.add_argument('--radius', type=float, required=True, \
        help='Radius of search area in degrees.')
    parser.add_argument('--database', type=str, required=True, \
        help='Database to search. Options: ztf, tess.')
    parser.add_argument('--start_date', type=str, required=True, \
        help='Start date of search in format YYYY-MM-DD.')
    parser.add_argument('--end_date', type=str, required=True, \
        help='End date of search in format YYYY-MM-DD.')
    args = parser.parse_args()

    # Call the function to retrieve the objects
    try:
        databases[args.database](args.ra, args.dec, args.radius, args.start_date, args.end_date)
    except Exception as e:
        print(f'Error retrieving objects: {e}')
