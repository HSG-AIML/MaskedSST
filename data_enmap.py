import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["GDAL_NUM_THREADS"] = "4"

import random
import glob
import pickle
from tqdm import tqdm
import numpy as np
import rasterio as rio
import warnings

import torch
from torch.utils.data import Dataset

wc_labels = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse vegetation",
    70: "Snow and Ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}
wc_labels_inv = {v: k for k,v in wc_labels.items()}

wc_labels_train = {
    0: "Tree cover",
    1: "Shrubland",
    2: "Grassland",
    3: "Cropland",
    4: "Built-up",
    5: "Bare/sparse vegetation",
    6: "Snow and Ice",
    7: "Permanent water bodies",
    8: "Herbaceous wetland",
    9: "Mangroves",
    10: "Moss and lichen",
}

dfc_labels = {
    1: "Forest",
    2: "Shrubland",
    3: "Savanna",
    4: "Grassland",
    5: "Wetlands",
    6: "Croplands",
    7: "Urban/Built-up",
    8: "Snow/Ice",
    9: "Barren",
    10: "Water",
}

dfc_labels_inv = {v: k for k,v in dfc_labels.items()}

dfc_labels_train = {
    0: "Forest",
    1: "Shrubland",
    2: "Grassland",
    3: "Wetlands",
    4: "Croplands",
    5: "Urban/Built-up",
    6: "Barren",
    7: "Water",
    -1: "Invalid",
}

wavelengths = [
    418.24 ,  423.874,  429.294,  434.528,  439.603,  444.549,
        449.391,  454.159,  458.884,  463.584,  468.265,  472.934,
        477.599,  482.265,  486.941,  491.633,  496.349,  501.094,
        505.87 ,  510.678,  515.519,  520.397,  525.313,  530.268,
        535.265,  540.305,  545.391,  550.525,  555.71 ,  560.947,
        566.239,  571.587,  576.995,  582.464,  587.997,  593.596,
        599.267,  605.011,  610.833,  616.737,  622.732,  628.797,
        634.919,  641.1  ,  647.341,  653.643,  660.007,  666.435,
        672.927,  679.485,  686.11 ,  692.804,  699.567,  706.401,
        713.307,  720.282,  727.324,  734.431,  741.601,  748.833,
        756.124,  763.472,  770.876,  778.333,  785.843,  793.402,
        801.01 ,  808.665,  816.367,  824.112,  831.901,  839.731,
        847.601,  855.509,  863.455,  871.433,  879.442,  887.478,
        895.537,  902.257,  903.617,  911.715,  911.872,  919.827,
        921.624,  927.951,  931.512,  936.082,  941.53 ,  944.217,
        951.677,  952.355,  960.495,  961.948,  968.638,  972.341,
        976.783,  982.851,  984.932,  993.083,  993.475, 1004.21 ,
       1015.05 , 1026.   , 1037.05 , 1048.19 , 1059.42 , 1070.74 ,
       1082.14 , 1093.62 , 1105.17 , 1116.79 , 1128.47 , 1140.2  ,
       1151.98 , 1163.81 , 1175.67 , 1187.56 , 1199.48 , 1211.42 ,
       1223.37 , 1235.34 , 1247.31 , 1259.3  , 1271.29 , 1283.29 ,
       1295.28 , 1307.27 , 1319.25 , 1331.22 , 1343.18 , 1355.13 ,
       1367.06 , 1378.96 , 1390.84 , 1461.46 , 1473.1  , 1484.69 ,
       1496.24 , 1507.75 , 1519.22 , 1530.64 , 1542.02 , 1553.36 ,
       1564.65 , 1575.9  , 1587.1  , 1598.26 , 1609.36 , 1620.43 ,
       1631.44 , 1642.41 , 1653.33 , 1664.2  , 1675.03 , 1685.8  ,
       1696.53 , 1707.2  , 1717.83 , 1728.4  , 1738.93 , 1749.4  ,
       1759.83 , 1939.44 , 1948.98 , 1958.49 , 1967.95 , 1977.37 ,
       1986.74 , 1996.07 , 2005.36 , 2014.61 , 2023.82 , 2032.99 ,
       2042.11 , 2051.19 , 2060.24 , 2069.24 , 2078.21 , 2087.13 ,
       2096.01 , 2104.86 , 2113.67 , 2122.44 , 2131.17 , 2139.87 ,
       2148.52 , 2157.15 , 2165.73 , 2174.28 , 2182.79 , 2191.27 ,
       2199.71 , 2208.12 , 2216.5  , 2224.84 , 2233.14 , 2241.42 ,
       2249.66 , 2257.86 , 2266.04 , 2274.18 , 2282.29 , 2290.37 ,
       2298.42 , 2306.44 , 2314.42 , 2322.37 , 2330.29 , 2338.19 ,
       2346.05 , 2353.88 , 2361.68 , 2369.45 , 2377.19 , 2384.9  ,
       2392.58 , 2400.23 , 2407.85 , 2415.45 , 2423.01 , 2430.55 ,
       2438.05 , 2445.53,
]
# empty L2 bands
invalid_l2_bands = [
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    True, True, True, True, True, True, True, True, True, True, True,
    True, True, True, True, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, True, True, True, True, True, True,
    True, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False,
]

class EnMAPWorldCoverDataset(Dataset):
    def __init__(self, path, img_transforms, label_transform, device, pixel_location_file=None, num_samples_per_class=None, patch_size=3, patch_offset=100, test=False, load_to_memory=False, target_type="worldcover", remove_bands=[], shuffle_samples=False, clip=(-200,10000), rgb_only=False): 
        super().__init__()
        """ if  pixel_location_file is not none, num_samples_per_class pixels will be read in order from the pixel_location_file
        """
        assert target_type in ["worldcover", "dfc", "unlabeled"], f"target_type needs to be either worldcover or dfc: {target_type=}"
        self.nodata = -32768
        self.invalid_band_idxs = [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
        139, 140, 160, 161, 162, 163, 164, 165, 166]
        if remove_bands:
            self.invalid_band_idxs.extend(remove_bands)
        self.path = path
        self.transforms = img_transforms
        self.label_transform = label_transform
        self.device = device
        self.load_to_memory = load_to_memory
        self.testset = test
        self.target_type = target_type
        self.num_samples_per_class = num_samples_per_class
        self.pixel_location_file = pixel_location_file
        self.patch_size = patch_size
        self.patch_offset = patch_offset # only start reading patches lower than that value in the file for each class
        self.shuffle_samples = shuffle_samples # shuffle samples in the list
        self.clip = clip
        self.rgb_only = rgb_only

        if self.pixel_location_file is not None:
            assert 0 < num_samples_per_class < 6172 # max for dfc MexicoCity
            with open(self.pixel_location_file, "rb") as handle:
                self.pixel_locations = pickle.load(handle)

            if self.shuffle_samples:
                for k in list(self.pixel_locations.keys()):
                 random.shuffle(self.pixel_locations[k])

            # all pixels remove beyond the required number
            for k in list(self.pixel_locations.keys()):
                locs = []
                while len(locs) != self.num_samples_per_class:
                    tup = self.pixel_locations[k].pop(self.patch_offset)
                    # don't use pixels at the border of the tile (to make sure they can be patchified later)
                    x,y = tup[1]
                    if (x > self.patch_size):
                        if (x < (64 - self.patch_size)):
                            if (y > self.patch_size):
                                if (y < (64 - self.patch_size)):
                                    locs.append(tup)
                self.pixel_locations[k] = locs

            self.load_from_pixel_location_file_to_memory()

            # for k,v in self.pixel_locations.items():
                # print(k, len(v), len(set([x[0] for x in v])), sorted(set([x[0] for x in v])))

            print(f"{self.pixel_location_file=}")
            print(f"Number of samples: {len(self.patches):,}")

        if self.testset:
            assert "test" in path
        else:
            assert "train" in path

        if self.target_type in ["worldcover", "unlabeled"]:
            self.enmap_files = glob.glob(os.path.join(path, "*", "*enmap.tif"))
        elif self.target_type == "dfc":
            self.enmap_files = glob.glob(os.path.join(path, "*enmap.tif"))

        self.target_files = [f.replace("enmap.tif", f"{target_type}_30m.tif") for f in self.enmap_files]

        if self.target_type == "unlabeled":
            self.target_files = None

        if self.pixel_location_file is None:
            print(f"{self.pixel_location_file=}")
            print(f"Number of tiles: {len(self.enmap_files):,}")

        if self.load_to_memory:
            self.imgs = self.load_imgs()
            self.labels = None if self.target_type == "unlabeled" else self.load_labels()

    def __len__(self):
        if self.pixel_location_file is None:
            return len(self.enmap_files)
        else:
            return len(self.patches)

    def load_imgs(self):
        imgs = []
        print("Loading imgs...")
        for idx in tqdm(range(len(self)), total=len(self)):
            img = self.load_img(self.enmap_files[idx])
            imgs.append(img)

        return torch.stack(imgs) 

    def load_img(self, path):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Dataset has no geotransform, gcps, or rpcs. The identity matrix be returned.")
            with rio.open(path, num_threads=4) as f:
                img = f.read([x for x in f.indexes if x-1 not in self.invalid_band_idxs])
        img = self.transforms(img)
        if self.rgb_only:
            img = img[[199, 150, 0]]
        return img

    def load_label(self, path):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Dataset has no geotransform, gcps, or rpcs. The identity matrix be returned.")
            with rio.open(path, num_threads=4) as f:
                label = f.read()[0]
        label = self.label_transform(label)
        return label

    def load_labels(self):
        labels = []
        print("Loading labels...")
        for idx in tqdm(range(len(self)), total=len(self)):
            label = self.load_label(self.target_files[idx])
            labels.append(label)

        return torch.stack(labels) 

    def load_from_pixel_location_file_to_memory(self):
        self.patches = []
        self.labels = []

        k = 0
        prev_file = ""
        for c, v in tqdm(self.pixel_locations.items()):
            for pixel_info in v:
                if pixel_info[0] != prev_file:
                    # load the new tif file, if necessary
                    img = self.load_img(pixel_info[0])
                x,y = pixel_info[1]

                self.patches.append(img[:, x-self.patch_size//2 : (x+self.patch_size//2)+1, y-self.patch_size//2 : (y+self.patch_size//2)+1])
                self.labels.append(c)
                prev_file = pixel_info[0]
                k += 1

    def _getitem_from_list(self, idx):
            img = self.patches[idx]
            label = self.labels[idx] if self.target_type != "unlabeled" else None

            if self.clip is not None:
                img = torch.clip(img, min=self.clip[0], max=self.clip[1])

            return {"img": img, "label": label, "idx": idx}

    def __getitem__(self, idx):

        if self.pixel_location_file is not None:
            return self._getitem_from_list(idx)

        sample = {"idx": idx}

        img = self.imgs[idx] if self.load_to_memory else self.load_img(self.enmap_files[idx])
        if self.clip is not None:
            img = torch.clip(img, min=self.clip[0], max=self.clip[1])
        sample["img"] = img

        if self.target_type != "unlabeled":
            sample["label"] = self.labels[idx] if self.load_to_memory else self.load_label(self.target_files[idx]) 

        return sample

class StandardizeEnMAP(object):
    def __init__(self, use_clipped=True):
        super().__init__()
        # band-wise mean/std values from enmap dataset with 5000 samples
        # note that no-data bands have been removed
        self.use_clipped = use_clipped # clipped at -200:10000
        
        self.stds = np.array([
            162.6627, 167.3498, 174.0467, 179.4796, 185.6742, 189.8679, 195.2760,
        199.5368, 202.9833, 207.2796, 210.8276, 213.9201, 217.8574, 222.7276,
        227.5687, 231.4658, 235.7060, 240.4992, 243.6970, 247.6054, 251.5138,
        253.1133, 254.3610, 257.5711, 261.4103, 266.5251, 271.2709, 276.1599,
        282.2839, 289.4594, 297.8670, 306.1092, 313.6166, 323.6903, 329.2886,
        335.6399, 343.0602, 350.2030, 356.1875, 361.9685, 369.2584, 372.5967,
        376.6030, 384.3568, 390.5516, 395.6874, 403.0118, 410.9719, 416.5706,
        420.4139, 425.3302, 422.6733, 414.0843, 400.4410, 395.4434, 392.7087,
        411.5232, 440.8382, 474.4985, 494.2166, 519.8517, 512.8397, 539.7839,
        543.6790, 545.6442, 551.5317, 553.2733, 558.2146, 558.1660, 565.2712,
        563.3632, 569.1724, 571.8886, 572.3160, 576.3212, 574.2841, 578.4584,
        578.3580, 578.5734, 570.5172, 565.5851, 555.8351, 560.4955, 530.7009,
        575.9377, 569.8962, 527.2006, 591.8754, 550.5439, 581.9211, 532.5313,
        545.1941, 519.3048, 536.0895, 522.7880, 553.1370, 523.9921, 560.4667,
        526.0469, 524.4449, 569.6095, 574.6349, 579.3176, 586.3593, 592.8836,
        596.7551, 601.8961, 607.8694, 606.0878, 603.7205, 594.8602, 554.5336,
        552.9652, 553.3881, 523.8065, 544.9163, 551.5933, 556.8553, 559.0011,
        560.0006, 565.6541, 572.3035, 576.8378, 575.7842, 579.3028, 585.6633,
        585.4780, 584.7961, 585.3471, 583.9915, 584.2288, 585.7958, 588.2384,
        593.2621, 596.6155, 598.2057, 598.7576, 602.2360, 599.8145, 599.3627,
        595.8557, 593.7195, 593.5035, 591.3936, 588.1282, 541.4083, 567.5096,
        530.6711, 473.2966, 482.5293, 542.0838, 562.7828, 560.5759, 525.3482,
        519.5247, 522.0259, 538.0348, 531.8460, 526.6088, 529.4458, 528.9507,
        527.8317, 528.7123, 528.3576, 524.7983, 524.0628, 520.1588, 520.3376,
        516.3593, 515.5829, 506.0370, 508.8064, 510.3922, 511.1852, 506.1446,
        495.6286, 485.0877, 478.0323, 469.8576, 465.7909, 463.2858, 454.2642,
        447.1139, 446.0413, 436.3925, 434.2588, 432.2688, 422.2162, 418.1860,
        425.8918, 439.6881, 435.6189, 436.7505, 432.7513, 436.9694, 435.7019,
        426.5519, 407.2591, 420.3028, 420.1241,# 418.6790, 388.1842 
        ])
        self.means = np.array([
            352.5609,  333.3610,  340.2626,  372.5361,  384.9103,  391.9105,
         408.5285,  423.7897,  431.0917,  447.2268,  455.9118,  460.8083,
         465.9260,  477.1165,  485.3457,  487.0605,  492.6469,  506.5779,
         512.4955,  530.4929,  558.3329,  580.5131,  604.0617,  634.2068,
         660.5106,  685.5066,  701.3801,  717.9675,  733.4211,  743.7900,
         749.4088,  747.5000,  742.3693,  750.7346,  748.7164,  750.9659,
         762.0033,  769.6491,  771.8550,  770.8033,  778.5869,  779.3125,
         784.7033,  793.0665,  790.2686,  783.1088,  786.0762,  789.9905,
         793.7773,  804.4603,  831.1147,  885.4575, 1000.9156, 1133.5142,
        1275.3567, 1411.0153, 1592.4755, 1764.0706, 1903.5284, 1982.2793,
        2066.7998, 1950.6227, 2120.2627, 2162.9077, 2182.9558, 2201.2981,
        2218.9919, 2250.1230, 2229.8833, 2267.7332, 2278.5293, 2318.0803,
        2340.2087, 2357.7881, 2386.7212, 2399.2358, 2424.7900, 2412.5693,
        2411.9226, 2493.9631, 2369.9905, 2320.9121, 2456.5247, 2255.6382,
        2537.1460, 2405.4597, 2302.2434, 2280.7241, 2404.2720, 2281.5303,
        2350.6677, 2165.9363, 2145.6230, 2401.9368, 2254.5112, 2508.7412,
        2283.2461, 2559.8342, 2326.3315, 2356.4165, 2614.4976, 2645.4668,
        2681.7217, 2720.2461, 2755.3840, 2780.7529, 2811.6262, 2843.0125,
        2846.1531, 2845.3792, 2825.9941, 2622.9834, 2573.6826, 2516.6255,
        2360.1306, 2511.1770, 2553.1350, 2570.5024, 2581.4568, 2602.1743,
        2660.6331, 2724.1099, 2764.0994, 2788.8181, 2833.0964, 2829.0667,
        1787.0166, 1834.9420, 1881.9000, 1919.4335, 1959.9739, 1998.7937,
        2036.5725, 2083.3330, 2115.6299, 2139.5959, 2156.5791, 2177.1924,
        2166.8918, 2166.8440, 2143.6807, 2116.9592, 2089.6619, 2059.2410,
        2025.3713, 1129.9553, 1235.0288, 1136.8088,  960.8275,  988.2421,
        1253.9103, 1320.4712, 1343.7548, 1247.8154, 1260.2351, 1264.5610,
        1336.4877, 1323.5834, 1330.6691, 1341.5212, 1360.4840, 1362.7653,
        1385.0291, 1382.7590, 1394.6189, 1395.1858, 1405.9006, 1407.5414,
        1419.7426, 1418.2772, 1417.0790, 1421.5658, 1437.9874, 1429.2711,
        1420.7791, 1372.4348, 1342.6814, 1293.9158, 1267.3453, 1229.8119,
        1225.3783, 1178.6477, 1162.2891, 1132.3986, 1117.5476, 1101.0409,
        1107.9352, 1058.4889, 1056.4049, 1046.7291, 1091.4504, 1053.5328,
        1067.5254, 1018.6949, 1048.0970, 1008.7510,  997.1100,  883.8514,
         946.4255,  896.9833, # 894.8805,  709.3281 
        ])
        self.stds_clipped = np.array([
           161.7797, 166.4923, 173.2070, 178.6476, 184.8614, 189.0772, 194.5019,
        198.7791, 202.2433, 206.5507, 210.1110, 213.2190, 217.1704, 222.0500,
        226.8945, 230.8085, 235.0647, 239.8631, 243.0757, 246.9955, 250.9018,
        252.5122, 253.7767, 257.0009, 260.8510, 265.9680, 270.7267, 275.6290,
        281.7622, 288.9429, 297.3573, 305.6061, 313.1273, 323.2141, 328.8069,
        335.1597, 342.5912, 349.7433, 355.7289, 361.5123, 368.8019, 372.1354,
        376.1523, 383.9102, 390.0922, 395.2179, 402.5520, 410.5370, 416.1395,
        419.9804, 424.8458, 422.1497, 413.5729, 399.9206, 394.8736, 392.0406,
        410.8041, 440.1284, 473.8127, 493.5960, 519.1700, 512.1140, 539.1360,
        543.0515, 545.0151, 550.8640, 552.6069, 557.5392, 557.4506, 564.5439,
        562.6359, 568.4659, 571.1841, 571.6334, 575.6355, 573.5917, 577.7658,
        577.6354, 577.8055, 569.7978, 564.7845, 554.9975, 559.7739, 529.9006,
        575.2086, 568.8273, 526.4230, 586.0789, 549.5334, 577.3044, 531.6224,
        541.9402, 517.6170, 535.2849, 521.8011, 552.3841, 523.0508, 559.7584,
        525.1950, 523.6639, 568.9671, 574.0002, 578.7081, 585.7531, 592.2718,
        596.1400, 601.2558, 607.1868, 605.3873, 602.9781, 594.0599, 553.7220,
        552.0200, 552.4072, 522.9050, 544.0708, 550.7480, 555.9966, 558.1478,
        559.1515, 564.8092, 571.4623, 575.9857, 574.8973, 578.3912, 584.7681,
        584.4791, 583.7912, 584.3331, 582.9597, 583.1682, 584.7412, 587.1702,
        592.1866, 595.5479, 597.1395, 597.6829, 601.1622, 598.7451, 598.2917,
        594.7834, 592.6533, 592.4607, 590.3673, 587.1089, 540.3095, 566.4152,
        529.5828, 472.2783, 481.4532, 540.9426, 561.7067, 559.4681, 524.1749,
        518.3386, 520.8269, 536.8630, 530.7251, 525.5065, 528.3529, 527.8439,
        526.7190, 527.6174, 527.2777, 523.6768, 522.9105, 518.9763, 519.1688,
        515.1671, 514.3968, 504.8156, 507.6041, 509.1981, 509.9974, 504.9572,
        494.4387, 483.8880, 476.8270, 462.0763, 453.0680, 445.9120, 444.8457,
        435.1906, 433.0219, 431.0330, 420.9812, 416.9478, 424.6466, 438.4350,
        434.3660, 435.5033, 431.5199, 435.7241, 434.4594, 425.3207, 406.0712,
        419.0532, 418.8356, 417.3653, 386.9013, 
        ])
        self.means_clipped = np.array([
            352.4827,  333.2847,  340.1877,  372.4603,  384.8348,  391.8360,
         408.4543,  423.7160,  431.0192,  447.1547,  455.8404,  460.7380,
         465.8566,  477.0475,  485.2768,  486.9928,  492.5804,  506.5115,
         512.4302,  530.4281,  558.2677,  580.4486,  603.9984,  634.1443,
         660.4487,  685.4448,  701.3190,  717.9075,  733.3618,  743.7309,
         749.3500,  747.4417,  742.3122,  750.6784,  748.6596,  750.9095,
         761.9487,  769.5973,  771.8047,  770.7548,  778.5395,  779.2672,
         784.6599,  793.0247,  790.2291,  783.0744,  786.0458,  789.9599,
         793.7462,  804.4276,  831.0806,  885.4403, 1000.8979, 1133.5010,
        1275.3557, 1411.0566, 1592.5586, 1764.1697, 1903.6259, 1982.3396,
        2066.8687, 1950.7272, 2120.3218, 2162.9543, 2182.9924, 2201.3606,
        2219.0476, 2250.1775, 2229.9670, 2267.8193, 2278.6155, 2318.1455,
        2340.2661, 2357.8169, 2386.7483, 2399.2651, 2424.8149, 2412.6233,
        2412.0107, 2493.9866, 2370.1160, 2321.0764, 2456.5537, 2255.7832,
        2537.1711, 2405.8049, 2302.3359, 2286.6277, 2404.4944, 2285.7703,
        2350.8086, 2168.7266, 2146.6328, 2401.9817, 2254.7969, 2508.7393,
        2283.4932, 2559.8147, 2326.4944, 2356.5161, 2614.4712, 2645.4370,
        2681.6770, 2720.1997, 2755.3354, 2780.7012, 2811.5696, 2842.9604,
        2846.0974, 2845.3208, 2825.9370, 2622.9485, 2573.7100, 2516.6345,
        2360.0938, 2511.0928, 2553.0437, 2570.4072, 2581.3608, 2602.0764,
        2660.5330, 2724.0063, 2763.9934, 2788.7036, 2832.9771, 2828.9531,
        1786.8857, 1834.8105, 1881.7677, 1919.2994, 1959.8373, 1998.6576,
        2036.4349, 2083.1946, 2115.4917, 2139.4580, 2156.4407, 2177.0537,
        2166.7544, 2166.7065, 2143.5437, 2116.8235, 2089.5300, 2059.1130,
        2025.2448, 1129.8196, 1234.8900, 1136.6765,  960.7148,  988.1201,
        1253.7693, 1320.3326, 1343.6133, 1247.6725, 1260.0917, 1264.4150,
        1336.3412, 1323.4421, 1330.5300, 1341.3826, 1360.3444, 1362.6241,
        1384.8904, 1382.6224, 1394.4781, 1395.0414, 1405.7542, 1407.3965,
        1419.5956, 1418.1315, 1416.9316, 1421.4199, 1437.8427, 1429.1276,
        1420.6371, 1372.2946, 1342.5419, 1293.7775, 1225.2437, 1178.5161,
        1162.1582, 1132.2686, 1117.4178, 1100.9078, 1107.8024, 1058.3575,
        1056.2738, 1046.5966, 1091.3153, 1053.3986, 1067.3915, 1018.5629,
        1047.9631, 1008.6176,  996.9788,  883.7271,  946.2933,  896.8477,
         894.7427,  709.1987
        ])
    
    def __call__(self, x):
        if self.use_clipped:
            return (x - self.means_clipped[:, np.newaxis, np.newaxis]) / self.stds_clipped[:, np.newaxis, np.newaxis]
        return (x - self.means[:, np.newaxis, np.newaxis]) / self.stds[:, np.newaxis, np.newaxis]

    def reverse(self, x):
        if self.use_clipped:
            return x * self.stds_clipped[:, np.newaxis, np.newaxis] + self.means_clipped[:, np.newaxis, np.newaxis] 
        return x * self.stds[:, np.newaxis, np.newaxis] + self.means[:, np.newaxis, np.newaxis]
    
class MaxNormalizeEnMAP(object):
    def __init__(self):
        super().__init__()
        # band-wise max values from enmap dataset with 5000 samples
        # note that no-data bands have been removed
        self.maxs = np.array([ 24266.,  23937.,  23599.,  23322.,  23047.,  22809.,  22578.,
        22365.,  22184.,  22001.,  21836.,  21682.,  21531.,  21380.,
        21238.,  21112.,  20979.,  20844.,  20723.,  20597.,  20463.,
        20348.,  20230.,  20109.,  19993.,  19806.,  19182.,  18686.,
        18380.,  18044.,  17918.,  17787.,  17149.,  16703.,  16927.,
        16712.,  16278.,  16381.,  16596.,  16722.,  16850.,  17179.,
        16878.,  16746.,  17066.,  17025.,  16699.,  16155.,  15692.,
        15896.,  17508.,  18037.,  17740.,  17176.,  17705.,  17594.,
        17460.,  17329.,  17213.,  17081.,  17075.,  17061.,  16897.,
        16890.,  16854.,  16848.,  16808.,  16761.,  16734.,  16692.,
        16668.,  16631.,  16594.,  16532.,  16496.,  16465.,  16521.,
        16446.,  16416.,  16468.,  16404.,  16394.,  16460.,  16411.,
        16413.,  16396.,  16360.,  16285.,  16379.,  16276.,  16377.,
        16308.,  16321.,  16336.,  16349.,  16285.,  16319.,  16240.,
        16287.,  16261.,  15917.,  15567.,  15370.,  15410.,  15326.,
        15558.,  15653.,  15965.,  16001.,  15979.,  16232.,  16222.,
        16189.,  16162.,  16156.,  16124.,  16094.,  16066.,  16045.,
        15812.,  15792.,  15772.,  15754.,  15720.,  15891.,  15707.,
        15669.,  15660.,  15651.,  15637.,  15612.,  15619.,
        15602.,  15587.,  15596.,  15588.,  15577.,  15570.,  15563.,
        15558.,  15560.,  15551.,  15542.,  15534.,  15524., 
        15422.,
        15414.,  15414.,  15414.,  15414.,  15413.,  15410.,  15405.,
        15403.,  15401.,  15399.,  15396.,  15395.,  15393.,  15389.,
        15387.,  15371.,  15369.,  15368.,  15368.,  15366.,  15364.,
        15363.,  15361.,  15360.,  15355.,  15355.,  15354.,  15352.,
        15349.,  15348.,  15346.,  15358.,  15356.,  15355.,  15354.,
        15352.,  15350.,  15353.,  15353.,  15351.,  15350.,  15348.,
        15346.,  15344.,  15339.,  15336.,  15333.,  15331.,  15328.,
        15327.,  15323.,  15322.,  15319.,  15317.,  15314.,  15315.])
    
    def __call__(self, x):
        return x / self.maxs[:, np.newaxis, np.newaxis]

    def reverse(self, x):
        return x * self.maxs[:, np.newaxis, np.newaxis]

class MaxNormalizeAllBandsSame(object):
    def __init__(self):
        super().__init__()
        self.max = np.array([25000.])

    def __call__(self, x):
        return x / self.max

    def reverse(self, x):
        return x * self.max
    
class ToTensor(object):
    def __init__(self):
        super().__init__()
    
    def __call__(self, x):
        return torch.Tensor(x).to(torch.float32)

class WorldCoverLabelTransform(object):
    def __init__(self):
        super().__init__()
        # map labels from 10-100 to 0-10 range 
        self.old_to_new_labels = {
            0: -1,
            10: 0,
            20: 1,
            30: 2,
            40: 3,
            50: 4,
            60: 5,
            70: 6,
            80: 7,
            90: 8,
            95: 9,
            100: 10,
        }
        self.new_to_old_labels = {v:k for k,v in self.old_to_new_labels.items()}
    
    def __call__(self, x):
        x = torch.tensor(x, dtype=torch.long)

        x[x == 100] = 11
        x[x == 90] = 10
        x = torch.div(x, 10, rounding_mode='floor') - 1

        return x

class DFCLabelTransform(object):
    def __init__(self):
        super().__init__()
        # remove unused labels 3,8 and start labels at 0 instead of 1
        self.old_to_new_labels = {
            1: 0,
            2: 1,
            3: -1,
            4: 2,
            5: 3,
            6: 4,
            7: 5,
            8: -1,
            9: 6,
            10: 7,
        }
        self.new_to_old_labels = {v: k for k,v in self.old_to_new_labels.items()}

    def __call__(self, x):
        x = torch.tensor(x, dtype=torch.long)

        x[x == 3] = 0
        x[x == 8] = 0
        x[x >= 3] -= 1
        x[x >= 8] -= 1
        x -= 1

        return x

    def reverse(self, x):
        # add +1 back
        x += 1
        return x

    