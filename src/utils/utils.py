import numpy as np
import torch
import argparse

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
def logging(message, path, mode="w+"):
    print(message)
    message += "\n"
    with open(path, mode) as file:
        file.write(message)
        