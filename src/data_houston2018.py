import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["GDAL_NUM_THREADS"] = "4"

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
import spectral.io.envi as envi
from einops import rearrange

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

labels = [
    "Unclassified",
    "Healthy grass",
    "Stressed grass",
    "Artificial turf",
    "Evergreen trees",
    "Deciduous trees",
    "Bare earth",
    "Water",
    "Residential buildings",
    "Non-residential buildings",
    "Roads",
    "Sidewalks",
    "Crosswalks",
    "Major thoroughfares",
    "Highways",
    "Railways",
    "Paved parking lots",
    "Unpaved parking lots",
    "Cars",
    "Trains",
    "Stadium seats",
]
classes = list(range(21))

wavelengths = [
    374.399994,
    388.700012,
    403.100006,
    417.399994,
    431.700012,
    446.100006,
    460.399994,
    474.700012,
    489.0,
    503.399994,
    517.700012,
    532.0,
    546.299988,
    560.599976,
    574.900024,
    589.200012,
    603.599976,
    617.900024,
    632.200012,
    646.5,
    660.799988,
    675.099976,
    689.400024,
    703.700012,
    718.0,
    732.299988,
    746.599976,
    760.900024,
    775.200012,
    789.5,
    803.799988,
    818.099976,
    832.400024,
    846.700012,
    861.099976,
    875.400024,
    889.700012,
    904.0,
    918.299988,
    932.599976,
    946.900024,
    961.200012,
    975.5,
    989.799988,
    1004.200012,
    1018.5,
    1032.800049,
    1047.099976,
]
bands = list(range(48))


class Houston2018Dataset(Dataset):
    def __init__(
        self,
        path,
        label_path,
        transforms=None,
        label_transforms=None,
        patch_size=8,
        test=False,
        fix_train_patches=True,
        drop_unlabeled=False,
        pixelwise=False,
        rgb_only=False,
    ):
        """if fix_train_patches: fix patches that will be used for training during initialization
        if False, patches will be drawn randomly from the training data."""
        super().__init__()
        self.path = path
        self.label_path = label_path
        self.transforms = transforms
        self.label_transforms = label_transforms
        self.patch_size = patch_size
        self.fix_train_patches = fix_train_patches
        self.drop_unlabeled = drop_unlabeled
        self.pixelwise = pixelwise
        self.rgb_only = rgb_only

        if self.fix_train_patches:
            assert not test

        # in `test` mode, return non-overlapping patches sequentially s.t. evetually the whole test set is covered
        # if `test` is false, patches are sampled randomly from the (training) dataset
        self.test = test

        self.img = self.load_data()
        self.label = self.load_label()

        if self.test:
            print("Loading Houston2018 test split")
            # cut the entire scene into the three test sections (i.e., without the training data) and patchify
            img_test1 = self.img[:, :, :596]
            img_test2 = self.img[:, :601, 596:2980]
            img_test3 = self.img[:, :, 2980:]

            label_test1 = self.label[:, :596]
            label_test2 = self.label[:601, 596:2980]
            label_test3 = self.label[:, 2980:]

            img_patches_sections = []

            img_patches = []
            label_patches = []
            for img_area, label_area in zip(
                [img_test1, img_test2, img_test3],
                [label_test1, label_test2, label_test3],
            ):
                assert (
                    img_area.shape[1] == label_area.shape[0]
                    and img_area.shape[2] == label_area.shape[1]
                )

                x_sub = img_area.shape[1] % self.patch_size
                y_sub = img_area.shape[2] % self.patch_size

                if x_sub != 0:
                    img_area = img_area[:, :-x_sub, :]
                    label_area = label_area[:-x_sub, :]
                if y_sub != 0:
                    img_area = img_area[:, :, :-y_sub]
                    label_area = label_area[:, :-y_sub]

                img_area = rearrange(
                    img_area,
                    "c (h p0) (w p1) -> (h w) c p0 p1",
                    p0=self.patch_size,
                    p1=self.patch_size,
                )
                label_area = rearrange(
                    label_area,
                    "(h p0) (w p1) -> (h w) p0 p1",
                    p0=self.patch_size,
                    p1=self.patch_size,
                )

                if self.drop_unlabeled:
                    # drop patches that contain no classified pixels
                    valid_idx = np.array(
                        [
                            1 if label_area[i, :, :].sum() != 0 else 0
                            for i in range(label_area.shape[0])
                        ],
                        dtype=bool,
                    )
                else:
                    valid_idx = np.ones(label_area.shape[0], dtype=bool)

                img_patches.extend(img_area[valid_idx])
                label_patches.extend(label_area[valid_idx])
                img_patches_sections.append(img_area[valid_idx].shape[0])

            self.img_patches = img_patches
            self.label_patches = label_patches
            self.img_patches_sections = img_patches_sections
            # self.img = None
            # self.label = None
        else:
            # remove test data
            print("Loading Houston 2018 training split")
            self.img = self.img[:, 601:, 596:2980]
            if self.fix_train_patches:
                x_sub = self.img.shape[1] % self.patch_size
                y_sub = self.img.shape[2] % self.patch_size

                if x_sub != 0:
                    self.img = self.img[:, :-x_sub, :]
                    self.label = self.label[:-x_sub, :]
                if y_sub != 0:
                    self.img = self.img[:, :, :-y_sub]
                    self.label = self.label[:, :-y_sub]
                self.img_patches = rearrange(
                    self.img,
                    "c (h p0) (w p1) -> (h w) c p0 p1",
                    p0=self.patch_size,
                    p1=self.patch_size,
                )
                self.label_patches = rearrange(
                    self.label,
                    "(h p0) (w p1) -> (h w) p0 p1",
                    p0=self.patch_size,
                    p1=self.patch_size,
                )

                if self.drop_unlabeled:
                    # drop patches that contain no classified pixels
                    valid_idx = np.array(
                        [
                            1 if self.label_patches[i, :, :].sum() != 0 else 0
                            for i in range(self.label_patches.shape[0])
                        ],
                        dtype=bool,
                    )
                else:
                    valid_idx = np.ones(self.label.shape[0], dtype=bool)

                self.img_patches = self.img_patches[valid_idx]
                self.label_patches = self.label_patches[valid_idx]

                # self.img = None
                # self.label = None

        labeled_idx = (self.label != -1).nonzero()
        valid_labeled_idx = (
            (labeled_idx[:, 0] >= self.patch_size // 2)
            & (labeled_idx[:, 0] + self.patch_size // 2 < self.label.shape[0])
            & (labeled_idx[:, 1] >= self.patch_size // 2)
            & (labeled_idx[:, 1] + self.patch_size // 2 < self.label.shape[1])
        )
        self.labeled_idx = labeled_idx[valid_labeled_idx]

    def load_data(self):
        header_file = os.path.join(self.path, "20170218_UH_CASI_S4_NAD83.hdr")
        spectral_file = os.path.join(self.path, "20170218_UH_CASI_S4_NAD83.pix")

        data = envi.open(header_file, spectral_file)
        data = data.read_bands(range(data.shape[-1]))  # select the bands
        data = data[:, :, :-2]  # remove non-hsi bands

        data = np.moveaxis(data, -1, 0)  # h,w,c to c,h,w
        data = self.transforms(data)

        # zero pad the 48 bands to 50
        data = F.pad(data, (0, 0, 0, 0, 0, 2), "constant", 0)

        if self.rgb_only:
            data = data[[47, 31, 15], :, :]

        return data

    def load_label(self):
        with rio.open(self.label_path) as f:
            label = f.read(
                out_shape=(int(f.count), int(f.height / 2), int(f.width / 2)),
                resampling=Resampling.nearest,
            ).squeeze()

        label = self.label_transforms(label).to(torch.long)
        return label

    def __len__(self):
        if (self.test and not self.pixelwise) or self.fix_train_patches:
            return len(self.img_patches)
        elif self.pixelwise:
            return self.labeled_idx.shape[0]
        else:
            # approximate number of patches necessary to cover the image
            return (self.img.shape[1] // self.patch_size) * (
                self.img.shape[2] // self.patch_size
            )

    def __getitem__(self, idx=None):
        if (self.test and not self.pixelwise) or self.fix_train_patches:
            # return fixed, non-overlapping patches that might not contain labeled pixels
            assert idx is not None
            return {"img": self.img_patches[idx], "label": self.label_patches[idx]}

        elif self.pixelwise:
            # return patches centered at labeled pixels
            x, y = self.labeled_idx[idx]
            if self.patch_size % 2 == 0:
                add = 0
            else:
                add = 1
            return {
                "img": self.img[
                    :,
                    x - self.patch_size // 2 : x + self.patch_size // 2 + add,
                    y - self.patch_size // 2 : y + self.patch_size // 2 + add,
                ],
                "label": self.label[x, y],
            }

        else:
            # sample patches at random locations from the data
            x = torch.randint(0, self.img.shape[1] - self.patch_size, size=(1,))
            y = torch.randint(0, self.img.shape[2] - self.patch_size, size=(1,))
            patch = self.img[:, x : x + self.patch_size, y : y + self.patch_size]
            label = self.label[x : x + self.patch_size, y : y + self.patch_size]

            if label.sum() == 0 and self.drop_unlabeled:
                return self.__getitem__()

            return {"img": patch, "label": label}


class StandardizeHouston2018(object):
    def __init__(self):
        super().__init__()
        # band-wise mean/std values from train split of Houston 2018 dataset

        self.stds = np.array(
            [
                147.40876032,
                310.05076922,
                683.3782606,
                1210.90961228,
                1530.31741683,
                1969.12243093,
                2366.66610463,
                2582.58238161,
                2621.8696138,
                2704.31436187,
                2663.14736274,
                2724.23966969,
                2673.92251701,
                2680.90856147,
                2698.80489257,
                2742.99238268,
                2709.30345676,
                2731.83394517,
                2656.18046927,
                2636.89687467,
                2611.5703655,
                2619.64653409,
                2341.44774173,
                2307.68135743,
                2173.04561167,
                2019.2331709,
                2160.23313815,
                1838.80421036,
                1924.80855217,
                2091.97114043,
                2036.23790679,
                1897.68830911,
                1813.67348897,
                1858.43368551,
                1762.59081529,
                1753.13293752,
                1768.15239215,
                1505.98453497,
                1354.97931153,
                1085.1820817,
                749.07066013,
                886.43487574,
                1244.91740708,
                1446.71474511,
                1512.6863924,
                1511.37039685,
                1506.48981591,
                1482.23713169,
            ]
        )
        self.means = np.array(
            [
                445.52755544,
                844.51918293,
                1493.19892531,
                2182.50867821,
                2519.90732658,
                3053.68866277,
                3526.56329984,
                3742.43938933,
                3721.03385437,
                3799.80596726,
                3779.9708358,
                3971.10421669,
                3971.36449248,
                3973.50642386,
                3923.58080981,
                3906.76690834,
                3818.84616593,
                3806.18407241,
                3673.47687788,
                3611.15493403,
                3537.63230675,
                3523.33611696,
                3216.4120991,
                3461.64604993,
                3710.78280118,
                3848.7909992,
                4296.56441725,
                3743.90529975,
                3936.42770229,
                4273.57874739,
                4177.22334699,
                3922.24596659,
                3776.09744805,
                3866.77102969,
                3675.25005653,
                3647.52095361,
                3674.91621068,
                3159.29767013,
                2854.15687361,
                2302.13930153,
                1622.44925613,
                1898.28560062,
                2618.67133916,
                3049.43461331,
                3205.78594331,
                3242.74881629,
                3259.34214997,
                3269.28198319,
            ]
        )

    def __call__(self, x):
        return (x - self.means[:, np.newaxis, np.newaxis]) / self.stds[
            :, np.newaxis, np.newaxis
        ]

    def reverse(self, x):
        return (
            x * self.stds[:, np.newaxis, np.newaxis]
            + self.means[:, np.newaxis, np.newaxis]
        )


class Houston2018LabelTransform(object):
    def __init__(self):
        """Move unclassified class 0 to -1
        valid range: 0-19"""
        super().__init__()

    def __call__(self, x):
        x = torch.tensor(x, dtype=torch.long)

        x -= 1

        return x
