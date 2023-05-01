# Cut EnMAP L2 tiles into patches and split into train and test set

import os
import glob
import pyproj
import rasterio
from tqdm import tqdm
from rasterio.warp import Resampling

from shapely.ops import transform

TILE_SIZE = 64
UPSCALE_FACTOR_ENMAP = 1  # 3 for 10m resolution with bilinear

ENMAP_PATH = "/ds2/remote_sensing/enmap/"
OUTPUT_DIR = "/ds2/remote_sensing/enmap_worldcover_dataset_v3/train/"
TESTFILES = "/ds2/remote_sensing/enmap_worldcover_dataset/testfiles.txt"

if __name__ == "__main__":
    wgs84 = pyproj.CRS("EPSG:4326")

    l2_product_dirs = [
        x
        for x in glob.glob(os.path.join(ENMAP_PATH, "*", "*", "*", "*L2A-DT*"))
        if os.path.isdir(x)
    ]
    l2_spectral_products = [
        glob.glob(os.path.join(d, "*SPECTRAL_IMAGE.TIF"))[0] for d in l2_product_dirs
    ]
    l2_metadata = [
        glob.glob(os.path.join(d, "*METADATA.XML"))[0] for d in l2_product_dirs
    ]
    print(f"Found {len(l2_spectral_products)} EnMAP products.")

    # make sure that there are no duplicate enmap files
    filenames = [x.split("/")[-1] for x in l2_spectral_products]
    assert len(filenames) == len(set(filenames))

    with open(TESTFILES) as f:
        testfiles = [x.strip() for x in f.readlines()]

    for enmap_product in l2_spectral_products:
        filename = enmap_product.split("/")[-1].split(".TIF")[0]
        outdir = os.path.join(OUTPUT_DIR, filename)

        if filename in testfiles:
            outdir = outdir.replace("train", "test")

        if os.path.exists(outdir):
            print("Directory already exists/EnMAP file already processed")
            save = False
            continue
        else:
            os.mkdir(outdir)
            save = True

        with rasterio.open(enmap_product) as dataset:
            # resample data to target shape
            enmap_meta = dataset.meta.copy()
            enmap_meta["bounds"] = dataset.bounds
            enmap = dataset.read(
                out_shape=(
                    dataset.count,
                    int(dataset.height * UPSCALE_FACTOR_ENMAP),
                    int(dataset.width * UPSCALE_FACTOR_ENMAP),
                ),
                resampling=Resampling.bilinear,
            )

            # scale image transform
            transform = dataset.transform * dataset.transform.scale(
                (dataset.width / enmap.shape[-1]), (dataset.height / enmap.shape[-2])
            )
            enmap_meta["transform"] = transform
            enmap_meta["width"] = enmap.shape[-1]
            enmap_meta["height"] = enmap.shape[-2]

        # create tiles from uncorrupted pixels
        tiles = []
        for i in range(0, enmap.shape[1], TILE_SIZE):
            for j in range(0, enmap.shape[2], TILE_SIZE):
                if i + TILE_SIZE > enmap.shape[1] or j + TILE_SIZE > enmap.shape[2]:
                    continue

                enmap_tile = enmap[:, i : i + TILE_SIZE, j : j + TILE_SIZE]

                if (enmap_tile == enmap_meta["nodata"]).mean(axis=(1, 2)).all():
                    # all bands are nodata for every pixel
                    continue

                tiles.append(enmap_tile)

        print(f"Number of valid tiles for {enmap_product.split('/')[-1]}: {len(tiles)}")

        if save:
            for idx, tile in tqdm(enumerate(tiles), total=len(tiles)):
                with rasterio.open(
                    os.path.join(outdir, f"tile{idx}_enmap.tif"),
                    "w",
                    driver="GTiff",
                    nodata=-32768.0,
                    dtype=tile.dtype,
                    count=tile.shape[0],
                    width=tile.shape[2],
                    height=tile.shape[1],
                ) as f:
                    f.write(tile)
        else:
            print("Not saved, see above")
