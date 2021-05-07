'''
How to run this script:
For example,

$ python run.py --model=resnet --experiment_title=resnet_balanced_16_1_batchnorm_true --epoch=1000 --class_a_weight=16 --class_b_weight=1 --use_batchnorm=1
'''
import numpy as np
import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn 
from sklearn.metrics import classification_report
import pickle
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# customized modules
from src.utils.utils import set_seed, set_arguments
from src.models.models import *
from src.training.training import *
from src.data.get_dataloaders import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    
    # 1. setting
    config = set_arguments()
    set_seed(config.seeds)
    print("Use CUDA: {}".format(torch.cuda.is_available()))
    
    # 2. read data
    loader_dict = create_dataloaders(
        data_dir=os.path.join(config.root, "data"), batch_size=config.batch_size,
        class_a_size=config.class_a_size, class_a_index=config.class_a_index,
        class_b_size=config.class_b_size, class_b_index=config.class_b_index,
        seeds=config.seeds, download_cifar10=config.download_cifar10
    )
    train_loader = loader_dict["train_loader"]
    test_ab_loader = loader_dict["test_ab_loader"]
    test_others_loader = loader_dict["test_others_loader"]
    
    # 3. train model
    # 3-1. training requirements
    if config.model == "resnet":
        model = ResNet(
            block=ResidualBlock,
            layers=[2, 2, 2],
            num_classes=config.num_classes,
            use_batchnorm=config.use_batchnorm
        ).to(device)
    elif config.model == "cnn":
        model = CustomCNN(config.num_classes).to(device)

    weights = torch.ones(config.num_classes)
    weights[config.class_a_index] = config.class_a_weight
    weights[config.class_b_index] = config.class_b_weight
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    learning_rate = config.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=config.l2_penalty)
    
    # 3-2. start training
    model_trained = train(model, train_loader, [test_ab_loader, test_others_loader], criterion, optimizer, config)
    
    # 3-3. save model checkpoints
    torch.save(
        model_trained.state_dict(),
        os.path.join(config.root, "results/models", config.experiment_title+".ckpt")
    )