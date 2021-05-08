# Importance-Weighting-in-Deep-Learning

## Summary

...

Note: 
[Google 
Doc 
Link](https://docs.google.com/document/d/1z7QGX-cHNsR0Ab-Gnr2Ra1oR9J0Q2-ldhoV6gBT1XxM/edit)

## Project Organization
--------

    ├── data
    │   ├── cifar-10-batches-py
    │   │   ├── batches.meta
    │   │   ├── data_batch_1
    │   │   ├── data_batch_2
    │   │   ├── data_batch_3
    │   │   ├── data_batch_4
    │   │   ├── data_batch_5
    │   │   ├── readme.html
    │   │   └── test_batch
    │   └── cifar-10-python.tar.gz
    ├── notebooks
    │   ├── pipeline.ipynb
    │   └── visualization.ipynb
    ├── README.md
    ├── requirements.txt
    ├── results
    │   ├── evaluation
    │   ├── fractions
    │   │   └── resnet_balanced_256_1_batchnorm_true.pkl
    │   │
    │   ├── logs
    │   │   └── resnet_balanced_256_1_batchnorm_true.txt
    │   │
    │   └── models
    │       └── resnet_balanced_256_1_batchnorm_true.ckpt
    │
    ├── run.ipynb
    ├── run.py
    └── src
        ├── __init__.py
        ├── data
        │   ├── download_data.py
        │   ├── get_dataloaders.py
        │   └── init__.py
        │
        ├── models
        │   └── models.py
        │
        ├── training
        │   └── training.py
        │
        └── utils
            └── utils.py

--------
