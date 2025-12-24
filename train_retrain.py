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
from utils.evaluation import Hook_handle, analysis # type: ignore
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Calculate volume of a Cylinder')
parser.add_argument('--setup', type=int, help='setups, 1:r18-c10, 2:r50-c100, 3:r18-ti, 4: vgg-ti')
parser.add_argument('--model_seed', type=int, default=1, help='model seed')
parser.add_argument('--device', type=int, default=0, help='device to use')
parser.add_argument("--decreasing_lr", default="91,136", help="decreasing strategy")
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--save_dir', type=str, default=None, help='directory to save the model')

parser.add_argument('--unlearn_seed', type=int, default=1, help='unlearn seed')

args = parser.parse_args()


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
    data_dir = data_dir # + '/cifar10'
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
    train_set = TinyImageNetDataset(train_set, cache_file='assets/tinyimagenet_preprocess/train.pt')
    test_set = ImageFolder(test_path, transform=test_transform)
    test_set = TinyImageNetDataset(test_set, cache_file='assets/tinyimagenet_preprocess/test.pt')

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)


forget_set_dir = f"assets/unlearn_set_idxs/{dataset}_forget_set_idx_{args.unlearn_seed}.pkl"
retain_set_dir = f"assets/unlearn_set_idxs/{dataset}_retain_set_idx_{args.unlearn_seed}.pkl"
with open(forget_set_dir, "rb") as f:
    fgt_set_idx = pickle.load(f)
with open(retain_set_dir, "rb") as f:
    rtn_set_idx = pickle.load(f)



model = model_dict[arch](num_classes=classes)
model.normalize = normalization

init_state = torch.load(f'assets/init_model/{dataset}_{arch}_init_weights_{model_init_seed}.pth', weights_only=True)
model.load_state_dict(init_state)

model = model.to(device)
hooker = Hook_handle()
hooker.set_hook(model, arch)



def _init_fn(worker_id):
    np.random.seed(int(model_init_seed))

forget_set = Subset(train_set, fgt_set_idx)
retain_set = Subset(train_set, rtn_set_idx)


det_forget_loader = DataLoader(
    forget_set,
    batch_size=batch_size,
    shuffle=False,
    worker_init_fn=_init_fn,
    num_workers=4,
    pin_memory=True
)

retain_loader = DataLoader(
    retain_set,
    batch_size=batch_size,
    shuffle=True,
    worker_init_fn=_init_fn,
    num_workers=4,
    pin_memory=True
)
det_retain_loader = DataLoader(
    retain_set,
    batch_size=batch_size,
    shuffle=False,
    worker_init_fn=_init_fn,
    num_workers=4,
    pin_memory=True
)


# train_loader = DataLoader(
#     train_set,
#     batch_size=batch_size,
#     shuffle=True,
#     worker_init_fn=_init_fn,
#     num_workers=4,
#     pin_memory=True
# )

# det_train_loader = DataLoader(
#     train_set,
#     batch_size=batch_size,
#     shuffle=False,
#     worker_init_fn=_init_fn,
#     num_workers=4,
#     pin_memory=True
# )

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
    name='retrain',
    group="retrain",
    config={
        "dataset": dataset,
        "arch": arch,
        "model_seed": model_init_seed,
        "unlearn_seed": args.unlearn_seed,
        "decreasing_lr": decreasing_lr,
        "unlearn": "Retrain",
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
    for x, y in tqdm(retain_loader, leave=False, desc=f"Epoch {epoch}"):
        x, y = x.to(device), y.to(device)
        logit = model(x)
        loss = criterion(logit, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    scheduler.step()
    wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)
    wandb.log({'avg_retain_loss': loss_sum / len(retain_loader)}, step=epoch)
    if epoch % 5 == 0 or epoch == epochs - 1:
        retain_loss, retain_accuracy, retain_NCC_mismatch, retain_norm_M_CoV, retain_norm_W_CoV, retain_Sw_invSb, retain_W_M_dist, retain_cos_M, retain_cos_W = analysis(model=model, criterion_summed=criterion, device=device, num_classes=classes, loader=det_retain_loader, hooker=hooker)
        test_loss, test_accuracy, test_NCC_mismatch, test_norm_M_CoV, test_norm_W_CoV, test_Sw_invSb, test_W_M_dist, test_cos_M, test_cos_W = analysis(model=model, criterion_summed=criterion, device=device, num_classes=classes, loader=test_loader, hooker=hooker)
        fgt_loss, fgt_accuracy, fgt_NCC_mismatch, fgt_norm_M_CoV, fgt_norm_W_CoV, fgt_Sw_invSb, fgt_W_M_dist, fgt_cos_M, fgt_cos_W = analysis(model=model, criterion_summed=criterion, device=device, num_classes=classes, loader=det_forget_loader, hooker=hooker)

        wandb.log({'retain_loss': retain_loss, 'retain_accuracy': retain_accuracy, 
                    'retain_NCC_mismatch': retain_NCC_mismatch, 'retain_norm_M_CoV': retain_norm_M_CoV, 'retain_norm_W_CoV': retain_norm_W_CoV, 
                    'retain_Sw_invSb': retain_Sw_invSb, 'retain_W_M_dist': retain_W_M_dist, 'retain_cos_M': retain_cos_M, 'retain_cos_W': retain_cos_W, 
                    'test_loss': test_loss, 'test_accuracy': test_accuracy,
                    'test_NCC_mismatch': test_NCC_mismatch, 'test_norm_M_CoV': test_norm_M_CoV, 'test_norm_W_CoV': test_norm_W_CoV,
                    'test_Sw_invSb': test_Sw_invSb, 'test_W_M_dist': test_W_M_dist, 'test_cos_M': test_cos_M, 'test_cos_W': test_cos_W,
                    'forget_loss': fgt_loss, 'forget_accuracy': fgt_accuracy,
                    'forget_NCC_mismatch': fgt_NCC_mismatch, 'forget_norm_M_CoV': fgt_norm_M_CoV, 'forget_norm_W_CoV': fgt_norm_W_CoV,
                    'forget_Sw_invSb': fgt_Sw_invSb, 'forget_W_M_dist': fgt_W_M_dist, 'forget_cos_M': fgt_cos_M, 'forget_cos_W': fgt_cos_W}, step=epoch)
        
        

# save model
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    
torch.save(model.state_dict(), os.path.join(args.save_dir, f'retrain_{dataset}_{arch}_model_{model_init_seed}_unlearn_{args.unlearn_seed}.pth'))
