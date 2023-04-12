# this script takes EnMAP products and their matched worldcover crops
# to create a dataset of smaller tiles
# see 01_match_enmap_wc_producty.py to match wc to enmap

import os
import numpy as np
import glob
from tqdm import tqdm
import rasterio
import rasterio.features
import rasterio.warp
import rasterio.mask

TILE_SIZE = 64
ENMAP_PATH = "/ds2/remote_sensing/enmap/"
OUTPUT_DIR = "/ds2/remote_sensing/enmap_worldcover_dataset/train/"
testfiles = "/ds2/remote_sensing/enmap_worldcover_dataset/testfiles.txt"

if __name__ == "__main__":
    with open(testfiles) as f:
        testfiles = [x.strip() for x in f.readlines()]

    l2_product_dirs = [x for x in glob.glob(os.path.join(ENMAP_PATH, "*", "*", "*", "*L2A-DT*")) if os.path.isdir(x)]
    l2_spectral_products = [glob.glob(os.path.join(d, "*SPECTRAL_IMAGE.TIF"))[0] for d in l2_product_dirs]
    l2_metadata = [glob.glob(os.path.join(d, "*METADATA.XML"))[0] for d in l2_product_dirs]
    print(f"Found {len(l2_spectral_products)} EnMAP products.")

    # make sure that there are no duplicate enmap files
    filenames = [x.split("/")[-1] for x in l2_spectral_products]
    assert len(filenames) == len(set(filenames))

    worldcover_files = [os.path.join(ENMAP_PATH, "esa_worldcover/tiles/", f) for f in filenames]

    for enmap_file, worldcover_file in zip(l2_spectral_products, worldcover_files):
        filename = enmap_file.split("/")[-1].split(".TIF")[0]
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

        with rasterio.open(enmap_file) as ef:
            enmap = ef.read()
            enmap_meta = ef.meta

        with rasterio.open(worldcover_file) as wcf:
            worldcover = wcf.read()
            worldcover_meta = wcf.meta

        tiles = []
        # print("Tiles with nodata values for all spectral bands:")
        for i in range(0, enmap.shape[1], TILE_SIZE):
            for j in range(0, enmap.shape[2], TILE_SIZE):
                if i+TILE_SIZE > enmap.shape[1] or j+TILE_SIZE > enmap.shape[2]:
                    continue
                    
                enmap_tile = enmap[:, i:i+TILE_SIZE, j:j+TILE_SIZE]
                
                if (enmap_tile == enmap_meta["nodata"]).mean(axis=(1,2)).all():
                    # all bands are nodata for every pixel
                    # print(f"{i}:{i+TILE_SIZE},{j}:{j+TILE_SIZE}")
                    continue
                    
                worldcover_tile = worldcover[0][i*3:i*3+TILE_SIZE*3, j*3:j*3+TILE_SIZE*3]
                if (worldcover_tile == worldcover_meta["nodata"]).sum() > 0:
                    # print(f"{i}:{i+TILE_SIZE},{j}:{j+TILE_SIZE}, worldcover")
                    continue
                
                tiles.append((enmap_tile, worldcover_tile))
            
            
        f"Number of valid tiles for {enmap_file.split('/')[-1]}: {len(tiles)}"

        if save:
            for idx, (t,wc_t) in tqdm(enumerate(tiles), total=len(tiles)):
                with rasterio.open(os.path.join(outdir, f"tile{idx}_enmap.tif"), "w", driver="GTiff", nodata=-32768.0, dtype=t.dtype, count=t.shape[0], width=t.shape[2], height=t.shape[1]) as f:
                    f.write(t)
                with rasterio.open(os.path.join(outdir, f"tile{idx}_worldcover.tif"), "w", driver="GTiff", nodata=0.0, dtype=wc_t.dtype, count=1, width=wc_t.shape[1], height=wc_t.shape[0]) as f:
                    f.write(np.expand_dims(wc_t, 0))
        else:
            print("Not saved, see above")