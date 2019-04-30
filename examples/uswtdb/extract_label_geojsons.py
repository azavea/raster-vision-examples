import os
import sys
import boto3
import shutil
import rasterio

from copy import copy
from shapely.geometry import box

# geopandas, tqdm and joblib must be pip-installed in 
# RV docker container
import geopandas as gpd
from tqdm import tqdm
from joblib import Parallel, delayed

# bucket where the imagery lives
bucket = 'rasterfoundry-production-data-us-east-1'

# the states in question and the years we have imagery from
states = [('ia', '2017'), ('ok', '2017'), ('tx', '2016')]


def create_labels_for_scene(uri, state, turbines):
    '''
    Given the uri for a tiff on s3, check whether or not it contains any 
    wind turbines. If it does, write off a geojson with those turbines and 
    upload that geojson to s3.
    '''
    s3 = boto3.client('s3')
    
    # uri of the tif on s3
    tif_uri = 's3://{}/{}'.format(bucket, uri)

    with rasterio.open(tif_uri) as d:
        data = d.bounds
        # the tifs have different crs' depending on which utm zone they fall 
        # into. we need to gather the crs so that we can reproject the 
        # wind turbine geodataframe to match it
        crs = int(d.crs['init'].split(':')[1])

    # bounding box of tif
    bbox = box(data[0], data[1], data[2], data[3])
    # reproject turbines to match crs of bbox
    turbines = turbines.to_crs(epsg=crs)
    # just turbines that intersect the bounding box
    tmp = turbines[turbines.intersects(bbox)]

    # if we do find turbines in this bounding box
    if len(tmp) > 0:
        # long/lat for use in raster vision
        tmp = tmp.to_crs({'init': 'epsg:4326'})

        # write the geojson of selected turbines to temporary directory
        geojson_file = os.path.basename(tif_uri).replace('.tif', '.geojson')
        outfile = os.path.join('tmp/', geojson_file)
        tmp.to_file(driver='GeoJSON', filename=outfile)

        # upload that geojson to specific location on s3
        geojson_bucket = 'raster-vision-wind-turbines'
        filename = 'labels/{}/{}'.format(state, geojson_file)
        s3.upload_file(outfile, geojson_bucket, filename)

        # remove the local geojson
        os.remove(outfile)


def main(point_geojson):
    s3 = boto3.client('s3')
    
    # create polygon geodataframe of wind turbines in selected states
    # read in point geojson file of wind turbines
    wind_turbines = gpd.read_file(point_geojson)
    # get the abbreviations for the states of interes
    selected_states = list(map(lambda x: x[0].upper(), states))
    # filter turbines in selected states
    selected_turbines = wind_turbines[
        wind_turbines.t_state.isin(selected_states)]
    # buffer with a radius of 50 m to create polygons for each wind turbine
    selected_turbines_poly = copy(selected_turbines)
    selected_turbines_poly.geometry = selected_turbines.to_crs(
        epsg=3857).buffer(50)

    # create a temporary directory for geojsons
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')

    # for each state
    for state, year in tqdm(states):

    	# all of the tifs are in a series of folders within `rgb`, get the 
    	# path to those folders
        prefix_dir = 'naip-visualization/{}/{}/100cm/rgb/'.format(state, year)
        # get all of these directories
        all_directories = s3.list_objects(
            Bucket=bucket, Prefix=prefix_dir, Delimiter='/')
        prefixes = [x['Prefix'] for x in all_directories['CommonPrefixes']]

        # `list_objects` has a maximum of 1000 objects so we will need to run 
        # a separate call for each folder of tifs. 

        # set up progress bar 
        pbar = tqdm(prefixes)
        description = 'Generating list of tifs in {}:'.format(state.upper())
        pbar.set_description(description)

        # find all the tifs for the entire state
        dir_files = []
        for prefix in pbar:
            tifs = s3.list_objects(Bucket=bucket, Prefix=prefix)
            dir_files += [x['Key'] for x in tifs['Contents']]

        # then create labels for each tif
        # progress bar
        pbar = tqdm(dir_files)
        description = 'Checking for wind turbines in {}: '.format(
            state.upper())
        pbar.set_description(description)

        # parallelize using joblib
        Parallel(n_jobs=-1)(delayed(create_labels_for_scene)
                            (d, state, selected_turbines_poly) for d in pbar)
        
    # remove the temporary directory where the geojsons lived
    shutil.rmtree('tmp')

if __name__ == "__main__":
    args = sys.argv
    main(args[1])
