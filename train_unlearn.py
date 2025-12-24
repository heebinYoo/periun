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
from utils.evaluation import Hook_handle, analysis, get_micro_eval, get_acc # type: ignore
from tqdm import tqdm
from utils.unlearn_method import RandomLabel, FineTune, NegGrad, GAGD, NearLabel, BS # type: ignore
import torch
import numpy as np
import random
import os
import pandas as pd


def run_analysis_and_log(tag, loader):
    acc = get_acc(loader, model, device)
    wandb.log({f'{tag}_accuracy': acc}, step=epoch)

seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
os.environ["PYTHONHASHSEED"] = str(seed)

parser = argparse.ArgumentParser(description='Calculate volume of a Cylinder')
parser.add_argument('--setup', type=int, help='setups, 1:r18-c10, 2:r50-c100, 3:r18-ti, 4: vgg-ti')
parser.add_argument('--model_seed', type=int, default=1, help='model seed')
parser.add_argument('--device', type=int, default=0, help='device to use')
parser.add_argument("--decreasing_lr", default='0', help="decreasing strategy")
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--save_dir', type=str, default=None, help='directory to save the model')

parser.add_argument('--unlearn_seed', type=int, default=1, help='unlearn seed')
parser.add_argument('--unlearn_param', type=float)
parser.add_argument('--lr', type=float, help='learning rate')

parser.add_argument('--method', type=str, default='finetune', help='unlearn method')
parser.add_argument('--saliency_ratio', type=float, default=None)


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

if args.saliency_ratio is None:
    if args.method == "randomlabel":
        if args.setup == 1:
            args.saliency_ratio = 0.1
        elif args.setup == 2:
            args.saliency_ratio = 0.3
        elif args.setup == 3:
            args.saliency_ratio = 0.3
        elif args.setup == 4:
            args.saliency_ratio = 0.3
    elif args.method == "randomlabel_salun":
        if args.setup == 1:
            args.saliency_ratio = 0.1
        elif args.setup == 2:
            args.saliency_ratio = 0.3
        elif args.setup == 3:
            args.saliency_ratio = 0.5
        elif args.setup == 4:
            args.saliency_ratio = 0.5
    elif args.method == "neggrad":
        if args.setup == 1:
            args.saliency_ratio = 0.7
        elif args.setup == 2:
            args.saliency_ratio = 0.7
        elif args.setup == 3:
            args.saliency_ratio = 0.5
        elif args.setup == 4:
            args.saliency_ratio = 0.3
    elif args.method == "GAGD":
        if args.setup == 1:
            args.saliency_ratio = 0.7
        elif args.setup == 2:
            args.saliency_ratio = 0.3
        elif args.setup == 3:
            args.saliency_ratio = 0.5
        elif args.setup == 4:
            args.saliency_ratio = 0.5



def get_hyperparams():
    hyper_df = pd.read_csv('assets/hyperparam/hyperparam.csv')
    if args.lr is not None:
        return '_hyperparam_search'
    
    filt = (hyper_df['setup'] == args.setup) & \
           (hyper_df['method'] == args.method) & \
           (hyper_df['saliency_ratio'] == args.saliency_ratio)
    
    row = hyper_df[filt]
    
    if row.empty:
        raise ValueError(f"No hyperparams found for combination: {args}")
    
    args.lr = float(row.iloc[0]['lr'])
    val = row.iloc[0]['unlearn_param']
    args.unlearn_param = float(val) if pd.notna(val) else None
    return ''

title = get_hyperparams()

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
    data_dir = data_dir  # + '/cifar10'
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
    det_train_set = CIFAR10(data_dir, train=True, transform=test_transform, download=False)
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)
    det_train_set.targets = np.array(det_train_set.targets)
elif dataset == "cifar100":
    classes = 100
    data_dir = data_dir  + '/cifar100'
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
    det_train_set = CIFAR100(data_dir, train=True, transform=test_transform, download=False)
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)
    det_train_set.targets = np.array(det_train_set.targets)
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
    det_train_set = ImageFolder(train_path, transform=test_transform)
    det_train_set = TinyImageNetDataset(det_train_set, cache_file='assets/tinyimagenet_preprocess/det_train.pt')
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)
    det_train_set.targets = np.array(det_train_set.targets)








model = model_dict[arch](num_classes=classes)
model.normalize = normalization

original_model_dir = f'assets/orig_model/{dataset}_{arch}_model_{model_init_seed}.pth'
orig_state = torch.load(original_model_dir, weights_only=True)
model.load_state_dict(orig_state)

model = model.to(device)
hooker = Hook_handle()
hooker.set_hook(model, arch)

forget_set_dir = f"assets/unlearn_set_idxs/{dataset}_forget_set_idx_{args.unlearn_seed}.pkl"
retain_set_dir = f"assets/unlearn_set_idxs/{dataset}_retain_set_idx_{args.unlearn_seed}.pkl"
with open(forget_set_dir, "rb") as f:
    fgt_set_idx = pickle.load(f)
with open(retain_set_dir, "rb") as f:
    rtn_set_idx = pickle.load(f)


def _init_fn(worker_id):
    np.random.seed(int(model_init_seed))


retain_set = Subset(train_set, rtn_set_idx)
det_retain_set = Subset(det_train_set, rtn_set_idx)
retain_loader = DataLoader(
    retain_set,
    batch_size=batch_size,
    shuffle=True,
    worker_init_fn=_init_fn,
    num_workers=4,
    pin_memory=True
)
det_retain_loader = DataLoader(
    det_retain_set,
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




### saliency masking
det_forget_loader = DataLoader(
    Subset(det_train_set, fgt_set_idx),
    batch_size=batch_size,
    shuffle=False,
    worker_init_fn=_init_fn,
    num_workers=4,
    pin_memory=True
)

confidence = torch.empty(len(fgt_set_idx))
idx = 0
with torch.no_grad():
    for x, _ in det_forget_loader:
        x = x.to(device)
        bs = x.size(0)
        logit = model(x)
        confidence[idx:idx+bs] = torch.softmax(logit, dim=1).max(dim=1).values.cpu()
        idx += bs

lowest_confidence_indices = torch.argsort(confidence)[:int(args.saliency_ratio * len(confidence))]



forget_set = Subset(train_set, fgt_set_idx[lowest_confidence_indices])
forget_loader = DataLoader(
    forget_set,
    batch_size=batch_size,
    shuffle=True,
    worker_init_fn=_init_fn,
    num_workers=4,
    pin_memory=True
)


###############################################




wandb.init(
    project="high_conf_delete",
    name='unlearn'+title,
    group=args.method,
    config={
        "dataset": dataset,
        "arch": arch,
        "model_seed": model_init_seed,
        "unlearn_seed": args.unlearn_seed,
        "decreasing_lr": decreasing_lr,
        "unlearn": args.method,
        "unlearn_param": args.unlearn_param,
        "lr": lr,
        "epochs": epochs, 
        "saliency_ratio": args.saliency_ratio
    }
)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    args.lr,
    momentum=0.9,
    weight_decay=5e-4,
)



if args.method == 'randomlabel':
    method = RandomLabel(forget_loader, retain_loader, optimizer, classes, model_init_seed, device)
elif args.method == 'randomlabel_salun':
    mask = torch.load(f'assets/salun_mask/salun_{dataset}_{arch}_modelseed_{model_init_seed}_unlearnseed_{args.unlearn_seed}_saliency_{args.unlearn_param}')
    method = RandomLabel(forget_loader, retain_loader, optimizer, classes, model_init_seed, device, salun=mask)
elif args.method == 'nearlabel':
    method = NearLabel(forget_loader, retain_loader, optimizer, classes, model_init_seed, device, model)
elif args.method == 'nearlabel_salun':
    mask = torch.load(f'assets/salun_mask/salun_{dataset}_{arch}_modelseed_{model_init_seed}_unlearnseed_{args.unlearn_seed}_saliency_{args.unlearn_param}')
    method = NearLabel(forget_loader, retain_loader, optimizer, classes, model_init_seed, device, model, salun=mask)
elif args.method == 'finetune':
    method = FineTune(forget_loader, retain_loader, optimizer, device)
elif args.method == 'finetune_l1':
    method = FineTune(forget_loader, retain_loader, optimizer, device, l1=True, l1_param=args.unlearn_param)
elif args.method == 'neggrad':
    method = NegGrad(forget_loader, retain_loader, optimizer, device)
elif args.method == 'GAGD':
    method = GAGD(forget_loader, retain_loader, optimizer, alpha=args.unlearn_param, device=device)
elif args.method == 'boundary_shrink':
    method = BS(forget_loader, retain_loader, optimizer, classes, model_init_seed, device, model)
elif args.method == 'boundary_shrink_salun':
    mask = torch.load(f'assets/salun_mask/salun_{dataset}_{arch}_modelseed_{model_init_seed}_unlearnseed_{args.unlearn_seed}_saliency_{args.unlearn_param}')
    method = BS(forget_loader, retain_loader, optimizer, classes, model_init_seed, device, model, salun=mask)
else:
    raise ValueError("unknown unlearn method")



for epoch in tqdm(range(epochs)):
    wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)
    model.train()
    loss_avg = method.run(model=model, etc={'cur_epoch': epoch, 'unlearn_epochs': epochs} if (args.method == 'finetune' or args.method == "finetune_l1") else {})
    
    wandb.log({'avg_retain_loss': loss_avg}, step=epoch)
    if epoch % 5 == 0 or epoch == epochs - 1:
        run_analysis_and_log("retain", det_retain_loader)
        run_analysis_and_log("test", test_loader)
        run_analysis_and_log("forget", det_forget_loader)


                    
        
        

# save model
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
if title == '':
    torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.method}_{dataset}_{arch}_model_{model_init_seed}_unlearn_{args.unlearn_seed}.pth'))
else:
    torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.method}_{dataset}_{arch}_model_{model_init_seed}_unlearn_{args.unlearn_seed}_lr_{args.lr}_unlearn_param_{args.unlearn_param}.pth'))
wandb.finish()
        

