# this script scans a data directory for unzipped enmap products,
# extracts their bounding geometry and matches it with ESA WorldCover
# products.
# If necessary, worldcover products are merged s.t. they cover the 
# entire enmap scene and reprojected to the enmap crs.
# They are then matched with the enmap product such that they align
# pixel-wise.
# These matching tiles are stored on disk.

import os
import glob
from tqdm import tqdm
import pandas as pd
import rasterio
import rasterio.features
import rasterio.warp
import rasterio.mask
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling
from rasterio.warp import reproject, Resampling, aligned_target
import pyproj
from shapely.geometry import box
import pyproj

from shapely.ops import transform
import rasterio.merge

ENMAP_PATH = "/ds2/remote_sensing/enmap/"
WORLDCOVER_LOCATION_FILE = "esa_worldcover_locations.csv"
UPSCALE_FACTOR_ENMAP = 1 # 3 for 10m resolution with bilinear
WORLDCOVER_TARGET_RESOLUTION = 10

if __name__ == "__main__":
    l2_product_dirs = [x for x in glob.glob(os.path.join(ENMAP_PATH, "*", "*", "*", "*L2A-DT*")) if os.path.isdir(x)]
    l2_spectral_products = [glob.glob(os.path.join(d, "*SPECTRAL_IMAGE.TIF"))[0] for d in l2_product_dirs]
    l2_metadata = [glob.glob(os.path.join(d, "*METADATA.XML"))[0] for d in l2_product_dirs]
    print(f"Found {len(l2_spectral_products)} EnMAP products.")

    # make sure that there are no duplicate enmap files
    filenames = [x.split("/")[-1] for x in l2_spectral_products]
    assert len(filenames) == len(set(filenames))

    worldcover_locations = pd.read_csv(WORLDCOVER_LOCATION_FILE)

    # match each enmap product to 1 or 2 worldcover products
    # todo: handle 3 and 4 worldcover products (should be rare)
    wgs84 = pyproj.CRS('EPSG:4326')
    enmap_worldcover_matches = {}

    for enmap_product in l2_spectral_products:
        with rasterio.open(enmap_product) as f:
            
            crs = pyproj.CRS(f.crs.get("init"))
            projection = pyproj.Transformer.from_crs(crs, wgs84, always_xy=True).transform
            
            bounds_wgs84 = transform(projection, box(*f.bounds))
            worldcover_files = set()
            for lng,lat in bounds_wgs84.exterior.coords:
                wc_filename = worldcover_locations[((worldcover_locations.lat_lower <= lat) & (worldcover_locations.lat_upper >= lat)) & ((worldcover_locations.lon_lower <= lng) & (worldcover_locations.lon_upper >= lng))].filename.item()
                worldcover_files.add(os.path.join(ENMAP_PATH, "esa_worldcover/src", wc_filename))
            enmap_worldcover_matches[enmap_product] = worldcover_files

    # check which of those worldcover files are available locally
    worldcover_local = [f for f in glob.glob(os.path.join(ENMAP_PATH, "esa_worldcover/src", "*.tif")) if not "reprojection" in f]
    print(f"WorldCover files missing locally:\n\n{[x for x in set().union(*enmap_worldcover_matches.values()) if x not in worldcover_local]}")

    # if necessary, merge worldcover products
    print("Start merging WorldCover products...")
    for enmap in tqdm(list(enmap_worldcover_matches.keys())):
        wcfiles = enmap_worldcover_matches[enmap]
        
        if len(wcfiles) == 1:
            # unpack single item set
            enmap_worldcover_matches[enmap], = wcfiles
            continue
            
        if len(wcfiles) != 2:
            raise NotImplementedError("More than two files need to be merged...")
        
        # EnMAP covers two WorldCover tiles
        wcfiles = sorted(list(wcfiles))
        merged_file_name = os.path.join(ENMAP_PATH, "esa_worldcover/src/", wcfiles[0].split("/")[-1].replace(".tif", "_") + wcfiles[1].split("/")[-1])
        
        if os.path.isfile(merged_file_name):
            print("\tskipped merging, file already available:")
            print("\t\t", merged_file_name)
            enmap_worldcover_matches[enmap] = merged_file_name
            continue
            
        with rasterio.open(wcfiles[0]) as file1:
            with rasterio.open(wcfiles[1]) as file2:
                assert file1.crs == file2.crs, f"products have different crs {file1.crs}, {file2.crs}"
                merged_data, merged_data_transform = rasterio.merge.merge([file1, file2])
                
            out_meta = file2.meta.copy()
            out_meta.update({
                "height": merged_data.shape[1],
                "width": merged_data.shape[2],
                "transform": merged_data_transform,
            })
            
            with rasterio.open(merged_file_name, "w", **out_meta) as out:
                out.write(merged_data)
                
        enmap_worldcover_matches[enmap] = merged_file_name

    enmap_worldcover_pairs = pd.DataFrame({"EnMAP": enmap_worldcover_matches.keys(), "WorldCover": enmap_worldcover_matches.values()})
    enmap_files_matched = [f.split("/")[-1] for f in glob.glob(os.path.join(ENMAP_PATH, "esa_worldcover/tiles", "*.TIF"))]
    enmap_files_not_matched = [f for f in l2_spectral_products if f.split("/")[-1] not in enmap_files_matched]
    enmap_worldcover_pairs["Processed"] = [False if enmap_file in enmap_files_not_matched else True for enmap_file in enmap_worldcover_pairs["EnMAP"]]

    print(f"Available EnMAP files: {enmap_worldcover_pairs.shape[0]}")
    print(f"Already processed: {enmap_worldcover_pairs['Processed'].sum()}")
    print(f"Remaining {(~enmap_worldcover_pairs['Processed']).sum()}")

    # drop already processed rows
    enmap_worldcover_pairs = enmap_worldcover_pairs[~enmap_worldcover_pairs['Processed']]

    for idx in tqdm(range(enmap_worldcover_pairs.shape[0]), total=enmap_worldcover_pairs.shape[0]):
        enmap_file = enmap_worldcover_pairs.iloc[idx]["EnMAP"]
        worldcover_file = enmap_worldcover_pairs.iloc[idx]["WorldCover"]

        worldcover_output_file = os.path.join(ENMAP_PATH, "esa_worldcover/tiles/", enmap_file.split("/")[-1])

        # read EnMAP
        with rasterio.open(enmap_file) as dataset:
            # resample data to target shape
            enmap_meta = dataset.meta.copy()
            enmap_meta["bounds"] = dataset.bounds
            enmap = dataset.read(
                out_shape=(
                    dataset.count,
                    int(dataset.height * UPSCALE_FACTOR_ENMAP),
                    int(dataset.width * UPSCALE_FACTOR_ENMAP)
                ),
                resampling=Resampling.bilinear
            )

            # scale image transform
            transform = dataset.transform * dataset.transform.scale(
                (dataset.width / enmap.shape[-1]),
                (dataset.height / enmap.shape[-2])
            )
            enmap_meta["transform"] = transform
            enmap_meta["width"] = enmap.shape[-1]
            enmap_meta["height"] = enmap.shape[-2]
        
        # reproject and crop worldcover

        # worldcover to enmap crs
        dst_crs = enmap_meta["crs"]

        with rasterio.open(worldcover_file) as src:
            worldcover_meta = src.meta.copy()
            
            transform, width, height = aligned_target(enmap_meta["transform"], enmap_meta["width"], enmap_meta["height"], resolution=WORLDCOVER_TARGET_RESOLUTION)
            
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            

            with rasterio.open(worldcover_file.split(".tif")[0] + "_reprojection.tif", 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest,window=None)
    
        # read from reprojected worldcover with EnMAP window
        # use windowed read instead of mask with polygon
        with rasterio.open(worldcover_file.split(".tif")[0] + "_reprojection.tif") as f:
            enmap_window = rasterio.windows.from_bounds(*enmap_meta["bounds"], f.transform)
            
            worldcover_window_transform = f.window_transform(enmap_window) # to write the window later
            worldcover = f.read(window=enmap_window)

        # write the matching worldcover product
        worldcover_out_meta = {
            "driver": "GTiff",
            "width": worldcover.shape[2],
            "height": worldcover.shape[1],
            "dtype": worldcover.dtype,
            "count": 1,
            "transform": worldcover_window_transform,
            "crs": enmap_meta["crs"],
            "nodata": 0.0,
        }

        with rasterio.open(worldcover_output_file, "w", **worldcover_out_meta) as f:
            f.write(worldcover[0], indexes=1)
