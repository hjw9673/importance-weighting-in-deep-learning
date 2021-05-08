# Importance-Weighting-in-Deep-Learning

## Summary

...

Note: 
[Google 
Doc 
Link](https://docs.google.com/document/d/1z7QGX-cHNsR0Ab-Gnr2Ra1oR9J0Q2-ldhoV6gBT1XxM/edit)

## Project Organization
--------

    ├── data                            <- used to save the downloaded cifar 10 dataset
    │   ├── cifar-10-batches-py
    │   │   ├── batches.meta
    │   │   ├── data_batch_1
    │   │   ├── data_batch_2
    │   │   ├── data_batch_3
    │   │   ├── data_batch_4
    │   │   ├── data_batch_5
    │   │   ├── readme.html
    │   │   └── test_batch
    │   └── cifar-10-python.tar.gz
    ├── notebooks                       <- used to save our experimenta jupyter notebooks
    │   ├── pipeline.ipynb
    │   └── visualization.ipynb
    ├── README.md
    ├── requirements.txt
    ├── results                         <- save any results generated during the experiments here.
    │   ├── evaluation                  <- save the pickle files of evaluation results, such as accuracy, classfication_result, .etc.
    │   ├── fractions                   <- save the pickle files of fractions o dogs on the test images here.
    │   ├── logs                        <- save the trainin logs/process as text files here.
    │   └── models                      <- save the trained model checkpoints.
    │
    ├── run.ipynb
    ├── run.py                          <- this is the main script for running the experiments.
    └── src                             <- production codes are separated from notebooks and are saved here.
        ├── __init__.py
        ├── data                        
        │   ├── download_data.py
        │   ├── get_dataloaders.py      <- contain the function and class for generating pytorch dataloader.
        │   └── init__.py
        │
        ├── models
        │   └── models.py               <- contain classes for cnn and resnet.
        │
        ├── training
        │   └── training.py             <- contain train an evaluation function.
        │
        ├── utils
        │   └── utils.py                <- contain any utils functions here.
        │
        └── config
            └── config.py               <- contains all th arguments an hyperparameters for one experiment.
            
--------
