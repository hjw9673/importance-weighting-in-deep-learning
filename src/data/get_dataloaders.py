import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

class CustomDataLoader:
    
    def __init__(self, data_name, class_a_size=None, class_b_size=None, seeds=123, download=True):
        """
        Arguments:
            class_a_size: The number of samples for class a (e.g. dog).
            class_b_size: The number of samples for class b (e.g. cat).
        """
        self.data_name = data_name
        self.class_a_size = class_a_size # e.g. 5000
        self.class_b_size = class_b_size # e.g. 5000
        self.seeds = seeds
        if self.data_name == "cifar10":
            self.classes = ['plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        elif self.data_name == "mnist":
            self.classes = list(range(10))
        self.download = download
        
    def sampling(self, indices, num, seeds=123):
        """
        Sample num samples from indices randomly. If num is a float and 0.0 <= num <= 1.0,
        we sample num percentage of samples from indices.
        """
        np.random.seed(self.seeds)
        if isinstance(num, float) and 0 <= num <= 1:
            size = int(num * len(indices))
            samples = np.random.choice(indices, size)
        elif isinstance(num, int) and num <= len(indices):
            size = num
            samples = np.random.choice(indices, size)
        elif num is None:
            samples = indices
        else:
            print("Please make sure 'num' is in the correct range")
        return samples       
        
    def get_subset_index(self, labels, targets):
        
        # get subsets
        sample_indices = []
        targets = set(targets)
        for i in range(len(labels)):
            if labels[i] in targets:
                sample_indices.append(i)
            
        return sample_indices

    def load_dataset(self, data_dir, batch_size, train_a_label=5, train_b_label=3):
        
        # 1. read train and test data by data_name
        if self.data_name == "cifar10":
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            trainset = torchvision.datasets.CIFAR10(
                root=data_dir, train=True,
                download=self.download, transform=transform
            )
            testset = torchvision.datasets.CIFAR10(
                root=data_dir, train=False,
                download=self.download, transform=transform
            )
        elif self.data_name == "mnist":
            transform = transforms.Compose(
                [transforms.Resize(32),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            trainset = torchvision.datasets.MNIST(
                root=data_dir, train=True,
                download=self.download, transform=transform
            )
            testset = torchvision.datasets.MNIST(
                root=data_dir, train=False,
                download=self.download, transform=transform
            )

        # 2. get index belonged to class a and b with sampling
        train_target_ls = trainset.targets.tolist() if torch.is_tensor(trainset.targets) else trainset.targets
        class_a_indices = self.get_subset_index(train_target_ls, [train_a_label])
        class_a_indices = self.sampling(class_a_indices, self.class_a_size, self.seeds)

        class_b_indices = self.get_subset_index(train_target_ls, [train_b_label])
        class_b_indices = self.sampling(class_b_indices, self.class_b_size, self.seeds)
        
        train_index_subset = np.concatenate((class_a_indices, class_b_indices)).tolist()
        trainsubset = torch.utils.data.Subset(trainset, train_index_subset)

        # 3. separate them into (dog, cat) and (the other 8)
        test_target_ls = testset.targets.tolist() if torch.is_tensor(testset.targets) else testset.targets
        test_index_class_ab = self.get_subset_index(
            labels=test_target_ls,
            targets=[train_a_label, train_b_label]
        )
        test_index_others = self.get_subset_index(
            labels=test_target_ls,
            targets=[cls for cls in range(len(self.classes)) if cls not in [train_a_label, train_b_label]]
        )
        testsubset_ab = torch.utils.data.Subset(testset, test_index_class_ab)
        testsubset_others = torch.utils.data.Subset(testset, test_index_others)

        return trainsubset, (testset, testsubset_ab, testsubset_others)

    def generate_loader(self, dataset, batch_size, shuffle=True, drop_last=True):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )
        return dataloader

    
def create_dataloaders(data_name, data_dir, batch_size, class_a_size, class_a_index,
                       class_b_size, class_b_index, seeds, download_cifar10):
    
    mydataloader = CustomDataLoader(
        data_name=data_name,
        class_a_size=class_a_size,
        class_b_size=class_b_size,
        seeds=seeds,
        download=download_cifar10
    )

    trainset, (testset, testset_ab, testset_others) = mydataloader.load_dataset(
        data_dir=data_dir,
        batch_size=batch_size,
        train_a_label=class_a_index, # dog
        train_b_label=class_b_index, # cat
    )

    train_loader = mydataloader.generate_loader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_loader = mydataloader.generate_loader(
        dataset=testset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_ab_loader = mydataloader.generate_loader(
        dataset=testset_ab,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_others_loader = mydataloader.generate_loader(
        dataset=testset_others,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    
    loaders = {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "test_ab_loader": test_ab_loader,
        "test_others_loader": test_others_loader,
    }
    
    return loaders