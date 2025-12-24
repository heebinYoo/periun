import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
def get_acc(loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for i, (image, target) in enumerate(tqdm(loader, leave=False)):
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    acc = correct / total
    return acc

def get_avg_mode_conf(loader, model, device, batch_size):
    with torch.no_grad():
        model.eval()
        confs = []
        uids = []
        for i, (image, target) in enumerate(tqdm(loader, leave=False, desc='get_avg_mode_conf')):
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            conf = F.softmax(output, dim=1).max(dim=1)[0].cpu()
            unique_id = range(i * batch_size, i * batch_size + len(conf))
            confs.append(conf)
            uids.append(unique_id)

        confs = torch.cat(confs)
        hist = torch.histc(confs, bins=50, min=confs.min().item(), max=confs.max().item())
        _, max_bin_idx = torch.max(hist, dim=0)
        bin_width = (confs.max() - confs.min()) / 50
        lower_bound = confs.min() + max_bin_idx * bin_width
        upper_bound = lower_bound + bin_width
        bin_center = (lower_bound + upper_bound) / 2
        avg_conf = confs.mean()
    return avg_conf.item(), bin_center.item(), confs.numpy(), uids

def get_acc_on_feature(loader, model, weight, bias, device):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for i, (image, target) in enumerate(tqdm(loader, leave=False)):
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            output = output @ weight.T + bias
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    acc = correct / total
    return acc

def get_micro_eval(loader, m1, m2, device):
    kl_sum = 0
    same_pred = 0
    total = 0
    with torch.no_grad():
        m1.eval()
        m2.eval()
        for i, (image, _) in enumerate(tqdm(loader, leave=False)):
            image = image.to(device)
            m1_output = m1(image)
            m2_output = m2(image)
            # KLD
            m1_log_prob = F.log_softmax(m1_output, dim=1)
            m2_log_prob = F.log_softmax(m2_output, dim=1)
            kl_sum += F.kl_div(m1_log_prob, m2_log_prob, reduction='sum', log_target=True).item()
            # agree
            _, m1_predicted = torch.max(m1_output, 1)
            _, m2_predicted = torch.max(m2_output, 1)
            same_pred += (m1_predicted == m2_predicted).sum().item()
            # total numver
            total += image.size(0)
    kld = kl_sum / total
    agreement = same_pred / total
    return kld, agreement

def get_micro_eval_seperate_correct(loader, m1, m2, device):
    same_correct = 0
    same_wrong = 0
    diff_m1_correct = 0
    diff_m2_correct = 0
    diff_all_wrong = 0    
    total = 0
    with torch.no_grad():
        m1.eval()
        m2.eval()
        for i, (image, label) in enumerate(tqdm(loader, leave=False)):
            image = image.to(device)
            label = label.to(device)
            m1_output = m1(image)
            m2_output = m2(image)
            # agree
            _, m1_predicted = torch.max(m1_output, 1)
            _, m2_predicted = torch.max(m2_output, 1)

            m1_correct = m1_predicted == label
            m2_correct = m2_predicted == label
            same_prediction = m1_predicted == m2_predicted
            same_correct_mask = same_prediction & m1_correct & m2_correct
            same_wrong_mask = same_prediction & (~m1_correct) & (~m2_correct)
            diff_m1_correct_mask = (~same_prediction) & m1_correct & (~m2_correct)
            diff_m2_correct_mask = (~same_prediction) & (~m1_correct) & m2_correct
            diff_all_wrong_mask = (~same_prediction) & (~m1_correct) & (~m2_correct)

            same_correct += same_correct_mask.sum().item()
            same_wrong += same_wrong_mask.sum().item()
            diff_m1_correct += diff_m1_correct_mask.sum().item()
            diff_m2_correct += diff_m2_correct_mask.sum().item()
            diff_all_wrong += diff_all_wrong_mask.sum().item()

            # total numver
            total += image.size(0)
    
    return same_correct/total, same_wrong/total, diff_m1_correct/total, diff_m2_correct/total, diff_all_wrong/total




class Hook_handle:
    def __init__(self):
        self.feature = None
        self.weight = None
        self.bias = None
        self.model_title = None
        self.feature_dim = None
    def set_hook(self, model, model_title):
        self.model_title = model_title
        def hook(module, input, output):
            self.feature = input[0].detach()
            self.weight = module.weight.detach()
            self.bias = module.bias.detach()
        if model_title == "vgg16_bn":
            model.classifier[4].register_forward_hook(hook)
            self.feature_dim = model.classifier[4].in_features
        else:
            model.fc.register_forward_hook(hook)
            self.feature_dim = model.fc.in_features
    def get_feature_dim(self):
        return self.feature_dim
        
    def get_feature(self):
        return self.feature
    def get_weight(self):
        return self.weight, self.bias
    
def analysis(model, criterion_summed, device, num_classes, loader, hooker):
    model.eval()
    C             = num_classes
    N             = [0 for _ in range(C)]
    mean          = [0 for _ in range(C)]
    Sw            = 0

    loss          = 0
    net_correct   = 0
    NCC_match_net = 0

    for computation in ['Mean','Cov']:
        for batch_idx, (data, target) in enumerate(loader, start=1):

            data, target = data.to(device), target.to(device)

            output = model(data)
            h = hooker.get_feature().view(data.shape[0],-1) # B CHW
            
            # during calculation of class means, calculate loss
            if computation == 'Mean':
                loss += criterion_summed(output, target).item()
                
            for c in range(C):
                # features belonging to class c
                idxs = (target == c).nonzero(as_tuple=True)[0]
                
                if len(idxs) == 0: # If no class-c in this batch
                  continue

                h_c = h[idxs,:] # B CHW

                if computation == 'Mean':
                    # update class means
                    mean[c] += torch.sum(h_c, dim=0) # CHW
                    N[c] += h_c.shape[0]
                    
                elif computation == 'Cov':
                    # update within-class cov

                    z = h_c - mean[c].unsqueeze(0) # B CHW
                    cov = torch.matmul(z.unsqueeze(-1), # B CHW 1
                                       z.unsqueeze(1))  # B 1 CHW
                    Sw += torch.sum(cov, dim=0)

                    # during calculation of within-class covariance, calculate:
                    # 1) network's accuracy
                    net_pred = torch.argmax(output[idxs,:], dim=1)
                    net_correct += sum(net_pred==target[idxs]).item()

                    # 2) agreement between prediction and nearest class center
                    NCC_scores = torch.stack([torch.norm(h_c[i,:] - M.T,dim=1) \
                                              for i in range(h_c.shape[0])])
                    NCC_pred = torch.argmin(NCC_scores, dim=1)
                    NCC_match_net += sum(NCC_pred==net_pred).item()

        
        if computation == 'Mean':
            for c in range(C):
                mean[c] /= N[c]
                M = torch.stack(mean).T
            loss /= sum(N)
        elif computation == 'Cov':
            Sw /= sum(N)
    accuracy = net_correct/sum(N)
    NCC_mismatch = 1-NCC_match_net/sum(N)

    # global mean
    muG = torch.mean(M, dim=1, keepdim=True) # CHW 1
    
    # between-class covariance
    M_ = M - muG
    Sb = torch.matmul(M_, M_.T) / C

    # avg norm
    W  = hooker.get_weight()[0]
    M_norms = torch.norm(M_,  dim=0)
    W_norms = torch.norm(W.T, dim=0)

    norm_M_CoV = (torch.std(M_norms)/torch.mean(M_norms)).item()
    norm_W_CoV = (torch.std(W_norms)/torch.mean(W_norms)).item()

    

    # tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    from scipy.sparse.linalg import svds
    eigvec, eigval, _ = svds(Sb, k=C-1)
    inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T 
    Sw_invSb = np.trace(Sw @ inv_Sb).item()

    # ||W^T - M_||
    normalized_M = M_ / torch.norm(M_,'fro')
    normalized_W = W.T / torch.norm(W.T,'fro')
    W_M_dist = (torch.norm(normalized_W - normalized_M)**2).item()

    # mutual coherence
    def coherence(V): 
        G = V.T @ V
        G += torch.ones((C,C),device=device) / (C-1)
        G -= torch.diag(torch.diag(G))
        return torch.norm(G,1).item() / (C*(C-1))

    cos_M = coherence(M_/M_norms)
    cos_W = coherence(W.T/W_norms)

    return loss, accuracy, NCC_mismatch, norm_M_CoV, norm_W_CoV, Sw_invSb, W_M_dist, cos_M, cos_W



def analysis_for_mixup(model, criterion_summed, device, num_classes, loader):
    model.eval()
    C             = num_classes
    N             = [0 for _ in range(C)]
    mean          = [0 for _ in range(C)]
    Sw            = 0

    loss          = 0
    net_correct   = 0
    NCC_match_net = 0
    for computation in ['Mean','Cov']:
        for batch_idx, (data, target) in enumerate(loader, start=1):
            data, target = data.to(device), target.to(device)

            output = model(data)
            h = output.view(data.shape[0],-1) # B CHW
            
            
            # during calculation of class means, calculate loss
            if computation == 'Mean':
                loss += criterion_summed(output, target).item()
                
            for c in range(C):
                # features belonging to class c
                idxs = (target == c).nonzero(as_tuple=True)[0]
                
                if len(idxs) == 0: # If no class-c in this batch
                  continue

                h_c = h[idxs,:] # B CHW

                if computation == 'Mean':
                    # update class means
                    mean[c] += torch.sum(h_c, dim=0) # CHW
                    N[c] += h_c.shape[0]
                    
                elif computation == 'Cov':
                    # update within-class cov

                    z = h_c - mean[c].unsqueeze(0) # B CHW
                    cov = torch.matmul(z.unsqueeze(-1), # B CHW 1
                                       z.unsqueeze(1))  # B 1 CHW
                    Sw += torch.sum(cov, dim=0)

                    # during calculation of within-class covariance, calculate:
                    # 1) network's accuracy
                    net_pred = torch.argmax((output @ criterion_summed.weights.T)[idxs,:], dim=1)
                    net_correct += sum(net_pred==target[idxs]).item()

                    # 2) agreement between prediction and nearest class center
                    NCC_scores = torch.stack([torch.norm(h_c[i,:] - M.T,dim=1) \
                                              for i in range(h_c.shape[0])])
                    NCC_pred = torch.argmin(NCC_scores, dim=1)
                    NCC_match_net += sum(NCC_pred==net_pred).item()

        
        if computation == 'Mean':
            for c in range(C):
                mean[c] /= N[c]
                M = torch.stack(mean).T
            loss /= sum(N)
        elif computation == 'Cov':
            Sw /= sum(N)
    accuracy = net_correct/sum(N)
    NCC_mismatch = 1-NCC_match_net/sum(N)

    # global mean
    muG = torch.mean(M, dim=1, keepdim=True) # CHW 1
    
    # between-class covariance
    M_ = M - muG
    Sb = torch.matmul(M_, M_.T) / C

    # avg norm
    W  = criterion_summed.weights
    M_norms = torch.norm(M_,  dim=0)
    W_norms = torch.norm(W.T, dim=0)

    norm_M_CoV = (torch.std(M_norms)/torch.mean(M_norms)).item()
    norm_W_CoV = (torch.std(W_norms)/torch.mean(W_norms)).item()

    

    # tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    from scipy.sparse.linalg import svds
    eigvec, eigval, _ = svds(Sb, k=C-1)
    inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T 
    Sw_invSb = np.trace(Sw @ inv_Sb).item()

    # ||W^T - M_||
    normalized_M = M_ / torch.norm(M_,'fro')
    normalized_W = W.T / torch.norm(W.T,'fro')
    W_M_dist = (torch.norm(normalized_W - normalized_M)**2).item()

    # mutual coherence
    def coherence(V): 
        G = V.T @ V
        G += torch.ones((C,C),device=device) / (C-1)
        G -= torch.diag(torch.diag(G))
        return torch.norm(G,1).item() / (C*(C-1))

    cos_M = coherence(M_/M_norms)
    cos_W = coherence(W.T/W_norms)

    return loss, accuracy, NCC_mismatch, norm_M_CoV, norm_W_CoV, Sw_invSb, W_M_dist, cos_M, cos_W