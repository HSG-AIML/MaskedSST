# Recreate the EnMAP dataset

Our work uses two datasets that are derived from EnMAP L2 data:
1. Large scale unsupervised EnMAP dataset
2. Supervised EnMAP-DFC dataset

To obtain the datasets, you first have to download the original source data from via the [EnMAP data portal](https://planning.enmap.org). This is possible after registration once your account has been approved for the `cat1distributor` role. The names of the data products used for our pre-training are listed in `enmap_products.txt`. The unsupervised dataset consists of `64x64` pixel patches and can be created with the `create_enmap_dataset.py` script.

To create the labeled EnMAP-DFC dataset, you also need a local version of the GRSS IADF Data Fusion Contest 2020 dataset. Set the path at the top of `create_enmap_dfc_dataset.py` to your local data and execute the script to create the dataset of matched patches.
