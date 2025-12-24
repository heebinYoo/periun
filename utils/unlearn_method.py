import torch
import copy
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from itertools import cycle
from tqdm import tqdm


class RandomLabel:
    def __init__(self, forget_loader, retain_loader, optimizer, classes, model_init_seed, device, salun=None):
        rand_label_dataset = copy.deepcopy(forget_loader.dataset.dataset)
        rand_label_dataset.targets = np.random.randint(0, classes, size=forget_loader.dataset.dataset.targets.shape)
        rand_label_fgt_set = Subset(rand_label_dataset, forget_loader.dataset.indices)
        self.dataloader = DataLoader(
            torch.utils.data.ConcatDataset([rand_label_fgt_set, retain_loader.dataset]),
            batch_size=forget_loader.batch_size,
            shuffle=True,
            worker_init_fn=lambda worker_id: np.random.seed(int(model_init_seed)),
            num_workers=4,
            pin_memory=True
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.salun = salun
        self.device = device
    def run(self, model, etc):
        model.train()
        loss_sum = 0
        for it, (image, target) in enumerate(tqdm(self.dataloader)):
            image = image.to(self.device)
            target = target.to(self.device)
            output_clean = model(image)
            loss = self.criterion(output_clean, target)
            self.optimizer.zero_grad()
            loss.backward()
            if self.salun:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= self.salun[name]
            self.optimizer.step()
            loss_sum += loss.item()
        return loss_sum / len(self.dataloader)
    

class BS:
    def __init__(self, forget_loader, retain_loader, optimizer, classes, model_init_seed, device, model, salun=None):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.salun = salun
        self.device = device
        self.bound = 0.1
        self.test_model = copy.deepcopy(model)
        self.test_model.eval()
        self.forget_loader = forget_loader

    def discretize(self, x):
        return torch.round(x * 255) / 255
    
    def run(self, model, etc):
        model.train()
        loss_sum = 0
        for it, (image, target) in enumerate(tqdm(self.forget_loader)):
            image = image.to(self.device)
            target = target.to(self.device)

            # BS attack
            x_adv = image.detach().clone().requires_grad_(True).to(self.device)
            pred = self.test_model(x_adv)
            loss = self.criterion(pred, target)
            loss.backward()
            grad_sign = x_adv.grad.data.detach().sign()
            x_adv = x_adv + grad_sign * self.bound
            x_adv = self.discretize(torch.clamp(x_adv, 0.0, 1.0)).detach()
            adv_label = torch.argmax(self.test_model(x_adv), dim=1)

            output_clean = model(image)
            loss = self.criterion(output_clean, adv_label)
            self.optimizer.zero_grad()
            loss.backward()
            if self.salun:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= self.salun[name]
            self.optimizer.step()
            loss_sum += loss.item()
        return loss_sum / len(self.forget_loader)

class NearLabel:
    def __init__(self, forget_loader, retain_loader, optimizer, classes, model_init_seed, device, model, salun=None):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.salun = salun
        self.device = device
    
        model.eval()
        det_train_loader = DataLoader(
            forget_loader.dataset.dataset,
            batch_size=forget_loader.batch_size,
            shuffle=False,
            worker_init_fn=lambda worker_id: np.random.seed(int(model_init_seed)),
            num_workers=4,
            pin_memory=True
        )
        new_targets = np.zeros((len(det_train_loader.dataset),), dtype=np.int64)
        with torch.no_grad():
            for it, (image, target) in enumerate(tqdm(det_train_loader, leave=False)):
                image = image.to(self.device)
                target = target.to(self.device)
                output_clean = model(image)
                top2 = torch.topk(output_clean, k=2, dim=1).indices
                del_target = torch.where(
                    top2[:, 0] != target,  # if top1 != del_target
                    top2[:, 0],                # use top1
                    top2[:, 1]                 # else use top2
                )
                new_targets[it * det_train_loader.batch_size: it * det_train_loader.batch_size + len(target)] = del_target.cpu().numpy()

        rand_label_dataset = copy.deepcopy(forget_loader.dataset.dataset)
        rand_label_dataset.targets = new_targets
        rand_label_fgt_set = Subset(rand_label_dataset, forget_loader.dataset.indices)
        self.dataloader = DataLoader(
            torch.utils.data.ConcatDataset([rand_label_fgt_set, retain_loader.dataset]),
            batch_size=forget_loader.batch_size,
            shuffle=True,
            worker_init_fn=lambda worker_id: np.random.seed(int(model_init_seed)),
            num_workers=4,
            pin_memory=True
        )


    def run(self, model, etc):
        model.train()
        loss_sum = 0
        for it, (image, target) in enumerate(tqdm(self.dataloader)):
            image = image.to(self.device)
            target = target.to(self.device)
            output_clean = model(image)
            loss = self.criterion(output_clean, target)
            self.optimizer.zero_grad()
            loss.backward()
            if self.salun:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= self.salun[name]
            self.optimizer.step()
            loss_sum += loss.item()
        return loss_sum / len(self.dataloader)
    

class FineTune:
    def __init__(self, forget_loader, retain_loader, optimizer, device, l1=False, l1_param=0):
        self.dataloader = retain_loader
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.l1 = l1
        self.l1_param = l1_param
        self.device = device
    def l1_regularization(self, model):
        params_vec = []
        for param in model.parameters():
            params_vec.append(param.view(-1))
        return torch.linalg.norm(torch.cat(params_vec), ord=1)
    def run(self, model, etc):
        cur_epoch = etc['cur_epoch']
        unlearn_epochs = etc['unlearn_epochs']
        loss_sum = 0
        model.train()
        current_alpha = self.l1_param * (1 - cur_epoch / (unlearn_epochs))

        for it, (image, target) in enumerate(tqdm(self.dataloader)):
            image = image.to(self.device)
            target = target.to(self.device)
            output_clean = model(image)
            loss = self.criterion(output_clean, target)
            if self.l1:
                loss += current_alpha * self.l1_regularization(model)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()
        return loss_sum / len(self.dataloader)
        
class NegGrad:
    def __init__(self, forget_loader, retain_loader, optimizer, device):
        self.dataloader = forget_loader
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.device = device
    def run(self, model, etc):
        model.train()
        loss_sum = 0
        for it, (image, target) in enumerate(tqdm(self.dataloader)):
            image = image.to(self.device)
            target = target.to(self.device)
            output_clean = model(image)
            loss = -self.criterion(output_clean, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()
        return loss_sum / len(self.dataloader)
class GAGD:
    def __init__(self, forget_loader, retain_loader, optimizer, alpha, device):
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.alpha = alpha
        self.device = device
    def run(self, model, etc):
        model.train()
        loss_sum = 0
        for idx, ((image, target), (del_input, del_target)) in enumerate(zip(self.retain_loader, cycle(self.forget_loader))):
            image, target, del_input, del_target = image.to(self.device), target.to(self.device), del_input.to(self.device), del_target.to(self.device)
            rtn_out = model(image)
            fgt_out = model(del_input)
            r_loss = self.criterion(rtn_out, target)
            del_loss = self.criterion(fgt_out, del_target)
            loss = self.alpha*r_loss - (1-self.alpha)*del_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()
        return loss_sum / len(self.retain_loader)
    


