import argparse
from ztfquery import query
from astropy.time import Time
from lightkurve import search_targetpixelfile
import pandas as pd

def get_ztf_data(ra, dec, radius, start_date):
    """Gets specified data from the ztf database. Only can access public data.

    Args:
        ra (float): right ascension
        dec (float): declination
        radius (float): radius of region of sky to search, with center at (ra, dec), in arcsec
        start_date (str): starting date to start search for
    """
    zquery = query.ZTFQuery()

    # convert start_date and end_date to MJD
    start_mjd = Time(start_date, format='iso').mjd

    zquery.load_metadata(radec=[ra,dec], size=radius, sql_query=f"fid=3 and obsjd>{start_mjd}")
    zquery.download_data("sciimg.fits", show_progress=True, nprocess=1, verbose=True, \
        overwrite=False)

def get_tess_data(ra, dec, radius, start_date):
    """Gets specified data from the tess database. Only can access public data.

    Args:
        ra (float): right ascension
        dec (float): declination
        radius (float): radius of region of sky to search, with center at (ra, dec), in arcsec
        start_date (str): starting date to start search for
    """
    target = str(ra) + " " + str(dec)
    pixelfile = search_targetpixelfile(target, radius=radius, mission='TESS').download()
    if pixelfile:
        pixelfile.download_all()
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
    args = parser.parse_args()

    # Call the function to retrieve the objects
    try:
        databases[args.database](args.ra, args.dec, args.radius, args.start_date)
    except Exception as e:
        print(f'Error retrieving objects: {e}')

known_supernovae = pd.read_csv('knownsn.csv')
ra = known_supernovae['RA']
dec = known_supernovae['DEC']

get_ztf_data(ra,dec,0.000277778,'2000-01-01')
