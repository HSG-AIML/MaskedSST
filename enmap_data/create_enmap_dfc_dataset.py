import os
import glob
import numpy as np
from tqdm import tqdm
import rasterio.merge
import rasterio as rio
from shapely.geometry import box

DFC_PATH = "/ds2/remote_sensing/grss-dfc-20"
ENMAP_PATH = "/ds2/remote_sensing/enmap"
# ENMAP_MEXICO_CITY_ID = "700424987"
ENMAP_MEXICO_CITY_IDS = [
    "ENMAP01-____L2A-DT0000006195_20221203T174715Z_003_V010111_20230103T090230Z",
    "ENMAP01-____L2A-DT0000006195_20221203T174720Z_004_V010111_20230103T074330Z",
]

OUTPUT_DIR = "/ds2/remote_sensing/enmap_dfc_dataset/MexicoCity_verification_run/"
TRAIN_TILES = os.path.join(OUTPUT_DIR, "train")
TEST_TILES = os.path.join(OUTPUT_DIR, "test")

TILE_SIZE = 64
TEST_IDS = "test_tile_ids.txt"


def downsample(data, factor=3):
    """downsample by keeping most frequent
    value in factorxfactor window
    """
    data_downsampled = np.zeros((data.shape[0] // factor, data.shape[1] // factor))
    for i in range(0, data_downsampled.shape[0]):
        for j in range(0, data_downsampled.shape[1]):
            v, c = np.unique(
                data[
                    factor * i : factor * i + factor, factor * j : factor * j + factor
                ],
                return_counts=True,
            )
            idx = np.argmax(c)
            data_downsampled[i, j] = v[idx]

    return data_downsampled


def merge_products(datasets):
    """
    merge adjacent rasterio datasets into a single combined tile
    """
    combined_datasets, combined_transform = rasterio.merge.merge(datasets)

    # get extent of combined dataset
    bounds_left = [d.bounds.left for d in datasets]
    bounds_right = [d.bounds.right for d in datasets]
    bounds_top = [d.bounds.top for d in datasets]
    bounds_bottom = [d.bounds.bottom for d in datasets]
    combined_bounds = rasterio.coords.BoundingBox(
        left=min(bounds_left),
        bottom=min(bounds_bottom),
        right=max(bounds_right),
        top=max(bounds_top),
    )

    # update metadata
    combined_meta = datasets[0].meta.copy()
    combined_meta.update(
        {
            "driver": "GTiff",
            "height": combined_datasets.shape[1],
            "width": combined_datasets.shape[2],
            "transform": combined_transform,
            "bounds": combined_bounds,
        }
    )

    return combined_datasets, combined_meta


if __name__ == "__main__":
    # get DFC Mexico City labels
    mc_dfc_files = glob.glob(
        os.path.join(
            DFC_PATH, "DFC_Public_Dataset/ROIs0000_winter/dfc_MexicoCity", "*.tif"
        )
    )
    print(f"Number of DFC files: {len(mc_dfc_files)}")

    datasets_dfc = [rio.open(file) for file in tqdm(mc_dfc_files)]
    assert len(set([d.crs for d in datasets_dfc])) == 1, "products have different crs"
    print(f"DFC CRS: {datasets_dfc[0].crs}")

    # merge DFC labels for Mexico City into single tile
    combined_dfc, combined_dfc_meta = merge_products(datasets_dfc)
    print(f"Shape of combined DFC products: {combined_dfc.shape}")

    # get the corresponding EnMAP data
    enmap_product_dirs = [
        x
        for x in glob.glob(os.path.join(ENMAP_PATH, "*", "*", "*", "*L2A-DT*"))
        if os.path.isdir(x)
    ]
    enmap_spectral_products = [
        glob.glob(os.path.join(d, "*SPECTRAL_IMAGE.TIF"))[0] for d in enmap_product_dirs
    ]

    enmap_mc_files = []
    for filename in ENMAP_MEXICO_CITY_IDS:
        enmap_mc_files.extend([f for f in enmap_spectral_products if filename in f])
    # enmap_mc_files = [f for f in enmap_spectral_products if f in ENMAP_MEXICO_CITY_IDS]
    print(f"EnMAP files for MexicoCity: {len(enmap_mc_files)}")

    enmap_mc_datasets = [rio.open(f) for f in tqdm(enmap_mc_files)]
    assert (
        len(set([d.crs for d in enmap_mc_datasets])) == 1
    ), "products have different crs"
    print(f"EnMAP CRS: {enmap_mc_datasets[0].crs}")

    # merge EnMAP products into single tile
    combined_mc_enmap, combined_enmap_meta = merge_products(enmap_mc_datasets)

    # save the combined products to disc
    with rio.open(
        os.path.join(OUTPUT_DIR, "enmap.tif"), "w", **combined_enmap_meta
    ) as f:
        f.write(combined_mc_enmap)

    # save the combined products to disc
    with rio.open(os.path.join(OUTPUT_DIR, "dfc.tif"), "w", **combined_dfc_meta) as f:
        f.write(combined_dfc)

    # match EnMAP with DFC labels
    assert (
        combined_dfc_meta["crs"] == combined_enmap_meta["crs"]
    ), "crs don't match, reproject EnMAP or DFC labels"

    with rio.open(os.path.join(OUTPUT_DIR, "enmap.tif")) as ef, rio.open(
        os.path.join(OUTPUT_DIR, "dfc.tif")
    ) as df:
        ext1 = box(*ef.bounds)
        ext2 = box(*df.bounds)
        intersection = ext1.intersection(ext2)

        win1 = rio.windows.from_bounds(*intersection.bounds, ef.transform)
        win2 = rio.windows.from_bounds(*intersection.bounds, df.transform)

        dfc_window_transform = df.window_transform(win1)
        enmap_window_transform = ef.window_transform(win2)

        enmap_matched = ef.read(window=win1)
        dfc_matched = df.read(window=win2)

        enmap_meta = ef.meta.copy()
        enmap_meta.update(
            {
                "width": enmap_matched.shape[2],
                "height": enmap_matched.shape[1],
                "transform": enmap_window_transform,
            }
        )
        dfc_meta = df.meta.copy()
        dfc_meta.update(
            {
                "width": dfc_matched.shape[2],
                "height": dfc_matched.shape[1],
                "transform": dfc_window_transform,
            }
        )

    # save the combined and matched products to disk
    with rio.open(
        os.path.join(OUTPUT_DIR, "enmap_matched.tif"), "w", **enmap_meta
    ) as f:
        f.write(enmap_matched)

    # save the combined and matched products to disk
    with rio.open(os.path.join(OUTPUT_DIR, "dfc_matched.tif"), "w", **dfc_meta) as f:
        f.write(dfc_matched)

    # create tiles
    tiles = []

    for i in range(0, enmap_matched.shape[1], TILE_SIZE):
        for j in range(0, enmap_matched.shape[2], TILE_SIZE):
            if (
                i + TILE_SIZE > enmap_matched.shape[1]
                or j + TILE_SIZE > enmap_matched.shape[2]
            ):
                continue

            enmap_tile = enmap_matched[:, i : i + TILE_SIZE, j : j + TILE_SIZE]
            if (enmap_tile == enmap_meta["nodata"]).mean(axis=(1, 2)).all():
                continue

            dfc_tile = dfc_matched[0][
                i * 3 : i * 3 + TILE_SIZE * 3, j * 3 : j * 3 + TILE_SIZE * 3
            ]
            if (dfc_tile == dfc_meta["nodata"]).sum() > 0:
                continue

            tiles.append((enmap_tile, dfc_tile))

    print(f"Number of valid tiles: {len(tiles)}")

    # sample test set
    # rng = rng = np.random.default_rng()
    # test_idx, = np.where((rng.random(len(tiles)) < TEST_FRAC) == 1)
    # print(f"Number of test tiles: {test_idx.shape}")

    with open(TEST_IDS) as f:
        test_idx = [int(x.strip()) for x in f.readlines()]

    # save tiles
    for idx, (t, dfc_t) in tqdm(enumerate(tiles), total=len(tiles)):
        outdir = TEST_TILES if idx in test_idx else TRAIN_TILES

        with rio.open(
            os.path.join(outdir, f"tile{idx}_enmap.tif"),
            "w",
            driver="GTiff",
            nodata=-32768.0,
            dtype=t.dtype,
            count=t.shape[0],
            width=t.shape[2],
            height=t.shape[1],
        ) as f:
            f.write(t)
        with rio.open(
            os.path.join(outdir, f"tile{idx}_dfc.tif"),
            "w",
            driver="GTiff",
            nodata=0.0,
            dtype=dfc_t.dtype,
            count=1,
            width=dfc_t.shape[1],
            height=dfc_t.shape[0],
        ) as f:
            f.write(np.expand_dims(dfc_t, 0))

    # create DFC labels at 30m resolution
    dfc_tiles_train = glob.glob(os.path.join(TRAIN_TILES, "*dfc.tif"))
    dfc_tiles_test = glob.glob(os.path.join(TEST_TILES, "*dfc.tif"))

    for i, dfc_file in tqdm(
        enumerate(dfc_tiles_train + dfc_tiles_test),
        total=len(dfc_tiles_train + dfc_tiles_test),
    ):
        low_res_file = dfc_file.replace(".tif", "_30m.tif")

        if os.path.exists(low_res_file):
            # don't read the file if the low-res version already exists on disk
            continue

        with rio.open(dfc_file) as dfc_dataset:
            dfc = dfc_dataset.read()
            dfc_downsampled = downsample(dfc[0])
            meta = dfc_dataset.meta.copy()

            meta["width"] = meta["width"] // 3
            meta["height"] = meta["height"] // 3

            with rio.open(low_res_file, "w", **meta) as f:
                f.write(np.expand_dims(dfc_downsampled, 0))
