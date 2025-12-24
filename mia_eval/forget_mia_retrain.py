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
from mia_eval.SVC_MIA import SVC_MIA
import csv



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

unlearn_seed = args.unlearn_seed
# for unlearn_seed in range(3):

train_loader, test_loader, forget_loader, retain_loader, normalization, classes = prepare_train_test_dataset(dataset, 0, batch_size, unlearn_seed)
model = model_dict[arch](num_classes=classes)
model.normalize = normalization
retrain_model = model
orig_model = copy.deepcopy(model)

data_dict = {
    "forget": forget_loader,
    # "retain": retain_loader,
    "test": test_loader,
}
test_len = len(test_loader.dataset)
forget_len = len(forget_loader.dataset)
retain_len = len(retain_loader.dataset)

subsampled_retain = Subset(retain_loader.dataset, np.random.choice(len(retain_loader.dataset), size=test_len, replace=False))
subsampled_retain_loader = DataLoader(subsampled_retain, batch_size=batch_size, shuffle=False, num_workers=4)
data_dict["subsampled_retain"] = subsampled_retain_loader


seed_dict = {}

for i in range(3):
    retrain_state = torch.load(f'assets/retrain_model/retrain_{dataset}_{arch}_model_{i}_unlearn_{unlearn_seed}.pth', weights_only=True)
    retrain_model.load_state_dict(retrain_state)
    retrain_model = retrain_model.to(device)
    retrain_model.eval()
    
    hook_retrain = Hook_handle()
    hook_retrain.set_hook(retrain_model, arch)

    feature_dim = hook_retrain.get_feature_dim()
    m = SVC_MIA(
        shadow_train=data_dict["subsampled_retain"],
        shadow_test=data_dict["test"],
        target_train=None,
        target_test=data_dict["forget"],
        model=model,
        device=device, classes=classes, hook=hook_retrain
    )
    csv_file_path = f'assets/mia/mia_results.csv'
    header = ['setup', 'unlearn_seed', 'init_seed', 'correctness_acc', 'confidence_acc', 'entropy_acc', 'm_entropy_acc', 'prob_acc']
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([
            args.setup,
            unlearn_seed,
            i,
            m['correctness'],
            m['confidence'],
            m['entropy'],
            m['m_entropy'],
            m['prob']
        ])

