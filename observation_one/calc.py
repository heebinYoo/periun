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
    logits = torch.empty((n, classes))
    features = torch.empty((n, feature_dim))
    norms = torch.empty(n)
    confidence = torch.empty(n)
    
    idx = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            batch_size = x.size(0)
            logit = model(x)
            feature = hook.get_feature()
            logits[idx:idx+batch_size] = logit.cpu()
            features[idx:idx+batch_size] = feature.cpu()
            norms[idx:idx+batch_size] = torch.norm(feature, dim=1, p=2).cpu()
            confidence[idx:idx+batch_size] = torch.softmax(logit, dim=1).max(dim=1).values.cpu()
            idx += batch_size
    return logits, features, norms, confidence


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
    
    retrain_state = torch.load(f'assets/retrain_model/retrain_{dataset}_{arch}_model_{i}_unlearn_{args.unlearn_seed}.pth', weights_only=True)
    retrain_model.load_state_dict(retrain_state)
    retrain_model = retrain_model.to(device)
    retrain_model.eval()
    
    hook_retrain = Hook_handle()
    hook_retrain.set_hook(retrain_model, arch)

    feature_dim = hook_retrain.get_feature_dim()
    

    orig_confidence_dict = {}
    retrain_confidence_dict = {}

    orig_feature_dict = {}
    retrain_feature_dict = {}

    for title, loader in data_dict.items():
        logits, features, norms, confidence = extract_model_outputs(
            orig_model, loader, feature_dim, device, classes, hook_orig
        )
        orig_confidence_dict[title] = confidence
        orig_feature_dict[title] = features

    for title, loader in data_dict.items():
        logits, features, norms, confidence = extract_model_outputs(
            retrain_model, loader, feature_dim, device, classes, hook_retrain
        )
        retrain_confidence_dict[title] = confidence
        retrain_feature_dict[title] = features
        
    seed_dict[i] = {
        "orig_confidence": orig_confidence_dict,
        "retrain_confidence": retrain_confidence_dict,
        "orig_feature": orig_feature_dict,
        "retrain_feature": retrain_feature_dict,
    }

os.makedirs(f"assets/figures/conf_diff", exist_ok=True)
os.makedirs(f"assets/figures/conf_diff/backdata", exist_ok=True)
import pickle
with open(f"assets/figures/conf_diff/backdata/{dataset}_{arch}_unlearn_seed_{args.unlearn_seed}_confidence_dict.pkl", "wb") as f:
    pickle.dump(seed_dict, f)




