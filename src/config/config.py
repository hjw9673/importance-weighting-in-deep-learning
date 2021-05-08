import argparse

def set_arguments():
    parser = argparse.ArgumentParser(description='Set up arguments for this experiment.')
    parser.add_argument(
        '--model', type=str,
        help='{cnn, resnet}')
    parser.add_argument(
        '--experiment_title', type=str,
        help='Set up the title of this experiment which will be the file name of model checkpoints and other pickle files.')
    parser.add_argument(
        '--root', default="./", type=str,
        help='The root path of this repository.')
    parser.add_argument(
        '--seeds', default=123, type=int,
        help='The random seed setting.')
    parser.add_argument(
        '--batch_size', default=16, type=int,
        help='The batch size of the trainloader.')
    parser.add_argument(
        '--class_a_index', default=5, type=int,
        help='The class index of the dog.')
    parser.add_argument(
        '--class_b_index', default=3, type=int,
        help='The class index of the cat.')
    parser.add_argument(
        '--class_a_size', default=None, type=float,
        help='The number of the dog samples.')
    parser.add_argument(
        '--class_b_size', default=None, type=float,
        help='The number of the cat samples.')
    parser.add_argument(
        '--class_a_weight', default=1, type=float,
        help='The root path of this repository.')
    parser.add_argument(
        '--class_b_weight', default=1, type=float,
        help='The root path of this repository.')
    parser.add_argument(
        '--epoch', default=1000, type=int,
        help='The number of training epochs.')
    parser.add_argument(
        '--download_cifar10', default=True, type=bool,
        help='Whether we need to download Cifar10 dataset from the website or not.')
    parser.add_argument(
        '--lr', default=0.1, type=float,
        help='The learning rate.')
    parser.add_argument(
        '--use_batchnorm', default=1, type=int,
        help='{1, 0}; Whether we apply Batchnorm after each convolutional layer or not.')
    parser.add_argument(
        '--num_classes', default=10, type=int,
        help='The number of class labels.')
    parser.add_argument(
        '--l2_penalty', default=0, type=float,
        help='The lambda value of l2 regularization.')
    args = parser.parse_args()

    return args