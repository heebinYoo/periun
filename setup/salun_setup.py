from torchvision import transforms
import sys
import os
from models import model_dict
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from dataset import TinyImageNetDataset
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import pickle
from itertools import cycle
import torch.nn as nn

class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)



device = torch.device(f"cuda:0")
# ,"cifar100", "TinyImagenet", "TinyImagenet"
# , "resnet50", "resnet18", "vgg16_bn"
for dataset, arch in zip(["cifar10", "cifar100", "TinyImagenet", "TinyImagenet"], ["resnet18", "resnet50", "resnet18", "vgg16_bn"]):
    data_dir = "tiny-imagenet-200" if dataset == "TinyImagenet" else "data"
    batch_size = 256

    if dataset == "cifar10":
        classes = 10
        data_dir = data_dir # +  '/cifar10'
        normalization = NormalizeByChannelMeanStd(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            )
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
        test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=False)
        train_set.targets = np.array(train_set.targets)
        test_set.targets = np.array(test_set.targets)
    elif dataset == "cifar100":
        classes = 100
        data_dir = data_dir # + '/cifar100'
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)
        test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=False)
        train_set.targets = np.array(train_set.targets)
        test_set.targets = np.array(test_set.targets)
    elif dataset == "TinyImagenet":
        classes = 200
        normalization = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
        test_transform = transforms.Compose([])
        train_path = os.path.join(data_dir, "train/")
        test_path = os.path.join(data_dir, "test/")
        train_set = ImageFolder(train_path, transform=train_transform)
        train_set = TinyImageNetDataset(train_set)
        test_set = ImageFolder(test_path, transform=test_transform)
        test_set = TinyImageNetDataset(test_set)
        train_set.targets = np.array(train_set.targets)
        test_set.targets = np.array(test_set.targets)

    for model_init_seed in range(3):
        for unlearn_data_seed in range(3):

            model = model_dict[arch](num_classes=classes)
            model.normalize = normalization

            orig_state = torch.load(f'assets/orig_model/{dataset}_{arch}_model_{model_init_seed}.pth', weights_only=True)
            model.load_state_dict(orig_state)
            model.cuda()
            def _init_fn(worker_id):
                np.random.seed(int(model_init_seed))

            ######## unlearn


            with open(f"assets/unlearn_set_idxs/{dataset}_forget_set_idx_{unlearn_data_seed}.pkl", "rb") as f:
                fgt_set_idx = pickle.load(f)
            forget_set = Subset(train_set, fgt_set_idx)
            forget_loader = DataLoader(
                forget_set,
                batch_size=batch_size,
                shuffle=True,
                worker_init_fn=_init_fn,
                num_workers=4,
                pin_memory=True
            )


            criterion = nn.CrossEntropyLoss()

            gradients = {}

            model.eval()

            for name, param in model.named_parameters():
                gradients[name] = 0

            for i, (image, target) in enumerate(forget_loader):
                image = image.cuda()
                target = target.cuda()
                # compute output
                output_clean = model(image)
                loss = -criterion(output_clean, target)
                loss.backward()

                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            gradients[name] += param.grad.data

            with torch.no_grad():
                for name in gradients:
                    gradients[name] = torch.abs_(gradients[name])

            threshold_list = [0.1, 0.2]

            for i in threshold_list:
                sorted_dict_positions = {}
                hard_dict = {}
                all_elements = torch.cat([tensor.flatten() for tensor in gradients.values()])
                threshold_index = int(len(all_elements) * i)
                positions = torch.argsort(all_elements)
                ranks = torch.argsort(positions)
                start_index = 0
                for key, tensor in gradients.items():
                    num_elements = tensor.numel()
                    tensor_ranks = ranks[start_index : start_index + num_elements]
                    sorted_positions = tensor_ranks.reshape(tensor.shape)
                    sorted_dict_positions[key] = sorted_positions
                    threshold_tensor = torch.zeros_like(tensor_ranks)
                    threshold_tensor[tensor_ranks < threshold_index] = 1
                    threshold_tensor = threshold_tensor.reshape(tensor.shape)
                    hard_dict[key] = threshold_tensor
                    start_index += num_elements
                
                import os
                save_dir = 'assets/salun_mask'
                os.makedirs(save_dir, exist_ok=True)
                torch.save(hard_dict, os.path.join(save_dir, f"salun_{dataset}_{arch}_modelseed_{model_init_seed}_unlearnseed_{unlearn_data_seed}_saliency_{i}"))
                    

