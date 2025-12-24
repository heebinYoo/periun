from torchvision import transforms
import sys
import os
from models import model_dict
from utils import NormalizeByChannelMeanStd
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from dataset import TinyImageNetDataset
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import pickle
from itertools import cycle
import argparse
import wandb
import torch.nn as nn
from microscopic_evaluation_expr.utils.evaluation import Hook_handle, analysis # type: ignore
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Calculate volume of a Cylinder')
parser.add_argument('--setup', type=int, help='setups, 1:r18-c10, 2:r50-c100, 3:r18-ti, 4: vgg-ti')
parser.add_argument('--model_seed', type=int, default=1, help='model seed')
parser.add_argument('--device', type=int, default=0, help='device to use')
parser.add_argument("--decreasing_lr", default="91,136", help="decreasing strategy")
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--save_dir', type=str, default=None, help='directory to save the model')


args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if args.setup == 1:
    arch = "resnet18"
    dataset = "cifar10"
elif args.setup == 2:
    arch = "resnet50"
    dataset = "cifar100"
elif args.setup == 3:
    arch = "resnet18"
    dataset = "TinyImagenet"
elif args.setup == 4:
    arch = "vgg16_bn"
    dataset = "TinyImagenet"
else:
    raise ValueError("setup must be 1, 2, 3 or 4")

data_dir = "tiny-imagenet-200" if dataset == "TinyImagenet" else "data"
model_init_seed = args.model_seed
batch_size = 256
if args.decreasing_lr != '0':
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
else:
    decreasing_lr = []
lr = args.lr
epochs = args.epochs
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")


if dataset == "cifar10":
    classes = 10
    data_dir = data_dir + '/cifar10'
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
    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=False)
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=False)
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)
elif dataset == "cifar100":
    classes = 100
    data_dir = data_dir + '/cifar100'
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
    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=False)
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

model = model_dict[arch](num_classes=classes)
model.normalize = normalization

init_state = torch.load(f'assets/init_model/{dataset}_{arch}_init_weights_{model_init_seed}.pth', weights_only=True)
model.load_state_dict(init_state)

model = model.to(device)
hooker = Hook_handle()
hooker.set_hook(model, arch)



def _init_fn(worker_id):
    np.random.seed(int(model_init_seed))

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    worker_init_fn=_init_fn,
    num_workers=4,
    pin_memory=True
)

det_train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=False,
    worker_init_fn=_init_fn,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    worker_init_fn=_init_fn,
    num_workers=4,
    pin_memory=True
)

wandb.init(
    project="microscopic_evaluation_expr",
    name='original',
    group="original",
    config={
        "dataset": dataset,
        "arch": arch,
        "model_seed": model_init_seed,
        "decreasing_lr": decreasing_lr,
        "lr": lr,
        "epochs": epochs
    }
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    args.lr,
    momentum=0.9,
    weight_decay=5e-4,
)

if dataset == "cifar100":
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.2
    )
elif dataset == "TinyImagenet" or dataset == "cifar10":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
else:
    raise ValueError("unknown dataset and architecture")

for epoch in tqdm(range(epochs)):
    model.train()
    loss_sum = 0
    for x, y in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}"):
        x, y = x.to(device), y.to(device)
        logit = model(x)
        loss = criterion(logit, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    scheduler.step()
    wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)
    wandb.log({'avg_train_loss': loss_sum / len(train_loader)}, step=epoch)
    if epoch % 5 == 0 or epoch == epochs - 1:
        train_loss, train_accuracy, train_NCC_mismatch, train_norm_M_CoV, train_norm_W_CoV, train_Sw_invSb, train_W_M_dist, train_cos_M, train_cos_W = analysis(model=model, criterion_summed=criterion, device=device, num_classes=classes, loader=det_train_loader, hooker=hooker)
        test_loss, test_accuracy, test_NCC_mismatch, test_norm_M_CoV, test_norm_W_CoV, test_Sw_invSb, test_W_M_dist, test_cos_M, test_cos_W = analysis(model=model, criterion_summed=criterion, device=device, num_classes=classes, loader=test_loader, hooker=hooker)
        wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy, 
                    'train_NCC_mismatch': train_NCC_mismatch, 'train_norm_M_CoV': train_norm_M_CoV, 'train_norm_W_CoV': train_norm_W_CoV, 
                    'train_Sw_invSb': train_Sw_invSb, 'train_W_M_dist': train_W_M_dist, 'train_cos_M': train_cos_M, 'train_cos_W': train_cos_W, 
                    'test_loss': test_loss, 'test_accuracy': test_accuracy,
                    'test_NCC_mismatch': test_NCC_mismatch, 'test_norm_M_CoV': test_norm_M_CoV, 'test_norm_W_CoV': test_norm_W_CoV,
                    'test_Sw_invSb': test_Sw_invSb, 'test_W_M_dist': test_W_M_dist, 'test_cos_M': test_cos_M, 'test_cos_W': test_cos_W}, step=epoch)
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'{dataset}_{arch}_model_{model_init_seed}_{epoch}.pth'))

        
