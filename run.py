'''
How to run this script:
For example,

$ python run.py --model=resnet --experiment_title=resnet_balanced_16_1_batchnorm_true --epoch=1000 --class_a_weight=16 --class_b_weight=1 --use_batchnorm=1 --l2_penalty=0
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

# customized modules
from src.utils.utils import set_seed
from src.config.config import set_arguments
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
    
    # 3-2. optimizer
    learning_rate = config.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=config.l2_penalty)
    
    # 3-3. retrain or initialize training
    if config.model_checkpoint != "none" and config.model_checkpoint.endswith(".ckpt"):
        print("Reload the model checkpoint: {}".format(config.model_checkpoint))
        checkpoint = torch.load(os.path.join(config.root, "results/models", config.model_checkpoint))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        reloaded_fractions = checkpoint['current_fraction_list']
    elif config.model_checkpoint == "none":
        reloaded_fractions = []
    
    # 3-4. loss function
    weights = torch.ones(config.num_classes)
    weights[config.class_a_index] = config.class_a_weight
    weights[config.class_b_index] = config.class_b_weight
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    
    # 3-5. start training
    model_trained = train(model, train_loader, [test_ab_loader, test_others_loader],
                          criterion, optimizer, config, reloaded_fractions)
    
    # 4. evaluate the model on the dog-cat testing images and save evaluation result
    evaluation_results = evaluation(model_trained, test_ab_loader, config)
    evaluation_file_path=os.path.join(config.root, "results/evaluation", config.experiment_title+".pkl")
    with open(evaluation_file_path, "wb") as file:
        pickle.dump(evaluation_results, file) 