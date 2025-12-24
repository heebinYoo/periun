from torchvision import transforms
import sys
import os
sys.path.append(os.path.abspath(".."))
from models import model_dict
from utils import NormalizeByChannelMeanStd
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from dataset import prepare_train_test_dataset
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import pickle
from itertools import cycle
from utils.evaluation import Hook_handle, analysis, get_micro_eval, get_acc, get_micro_eval_seperate_correct
import pandas as pd
import argparse
import random
import copy
from types import SimpleNamespace
import seaborn as sns
import matplotlib.pyplot as plt



def extract_model_outputs(model, loader, feature_dim, device, classes, hook):
    n = len(loader.dataset)
    softmaxs = torch.empty((n, classes))
    labels = torch.empty(n, dtype=torch.long)
    
    idx = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            batch_size = x.size(0)
            logit = model(x)
            softmaxs[idx:idx+batch_size] = torch.softmax(logit, dim=1).cpu()
            labels[idx:idx+batch_size] = y.cpu()
            idx += batch_size
    return softmaxs, labels


seed=42
random.seed(seed)

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
os.environ["PYTHONHASHSEED"] = str(seed)

parser = argparse.ArgumentParser(description='Calculate volume of a Cylinder')
parser.add_argument('--setup', type=int, default=1, help='setups, 1:r18-c10, 2:r50-c100, 3:r18-ti, 4: vgg-ti')
parser.add_argument('--unlearn_seed', type=int, default=1, help='unlearn seed')
parser.add_argument('--method', type=str, default='retrain', help='unlearning method')
parser.add_argument('--type', type=str)
args = parser.parse_args()
# args = SimpleNamespace(setup=2, unlearn_seed=0)



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

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 256

train_loader, test_loader, forget_loader, retain_loader, normalization, classes = prepare_train_test_dataset(dataset, 0, batch_size, args.unlearn_seed)
model = model_dict[arch](num_classes=classes)
model.normalize = normalization
retrain_model = model
orig_model = copy.deepcopy(model)

data_dict = {
    "forget": forget_loader,
    "retain": retain_loader,
    "train": train_loader,
    "test": test_loader,
}

seed_dict = {}

for i in range(3):
    orig_state = torch.load(f'assets/orig_model/{dataset}_{arch}_model_{i}.pth', weights_only=True)
    orig_model.load_state_dict(orig_state)
    orig_model = orig_model.to(device)
    orig_model.eval()

    hook_orig = Hook_handle()
    hook_orig.set_hook(orig_model, arch)
    
    if args.method == 'retrain':
        retrain_state = torch.load(f'assets/retrain_model/retrain_{dataset}_{arch}_model_{i}_unlearn_{args.unlearn_seed}.pth', weights_only=True)
    else:
        retrain_state = torch.load(f'assets/optimal_{args.type}_unlearn_model/{args.method}_{dataset}_{arch}_model_{i}_unlearn_{args.unlearn_seed}.pth', weights_only=True)
    retrain_model.load_state_dict(retrain_state)
    retrain_model = retrain_model.to(device)
    retrain_model.eval()
    
    hook_retrain = Hook_handle()
    hook_retrain.set_hook(retrain_model, arch)

    feature_dim = hook_retrain.get_feature_dim()
    
    orig_softmaxs_dict = {}
    retrain_softmaxs_dict = {}


    orig_label_dict = {}
    retrain_label_dict = {}

    for title, loader in data_dict.items():
        softmaxs, labels = extract_model_outputs(
            orig_model, loader, feature_dim, device, classes, hook_orig
        )
        orig_softmaxs_dict[title] = softmaxs
        orig_label_dict[title] = labels

    for title, loader in data_dict.items():
        softmaxs, labels = extract_model_outputs(
            retrain_model, loader, feature_dim, device, classes, hook_retrain
        )
        retrain_softmaxs_dict[title] = softmaxs
        retrain_label_dict[title] = labels
        
    seed_dict[i] = {
        "orig_softmax": orig_softmaxs_dict,
        "retrain_softmax": retrain_softmaxs_dict,
        "orig_label": orig_label_dict,
        "retrain_label": retrain_label_dict,
    }

import pickle
os.makedirs(f"assets/softmaxspace", exist_ok=True)
os.makedirs(f"assets/softmaxspace/backdata", exist_ok=True)
if args.method == 'retrain':
    os.makedirs(f"assets/softmaxspace/backdata/retrain/", exist_ok=True)
    with open(f"assets/softmaxspace/backdata/retrain/{dataset}_{arch}_unlearn_seed_{args.unlearn_seed}_confidence_change.pickle", "wb") as f:
        pickle.dump(seed_dict, f)
else:
    os.makedirs(f"assets/softmaxspace/backdata/{args.type}/", exist_ok=True)
    with open(f"assets/softmaxspace/backdata/{args.type}/{args.method}_{dataset}_{arch}_unlearn_seed_{args.unlearn_seed}_confidence_change.pickle", "wb") as f:
        pickle.dump(seed_dict, f)

