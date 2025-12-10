'''
STRIP: A Defence Against Trojan Attacks on Deep Neural Networks
This file is modified based on the following source:
link : https://github.com/Unispac/Fight-Poison-With-Poison/blob/master/other_cleansers/strip.py
This file implements the (detection) defense method called STRIP.

@inproceedings{gao2019strip,
    title={Strip: A defence against trojan attacks on deep neural networks},
    author={Gao, Yansong and Xu, Change and Wang, Derui and Chen, Shiping and Ranasinghe, Damith C and Nepal, Surya},
    booktitle={Proceedings of the 35th Annual Computer Security Applications Conference},
    pages={113--125},
    year={2019}}

basic sturcture for defense method:

STRIP detection:
    a. mix up clean samples and record the entropy of prediction, and record smallest entropy and largest entropy as thresholds.
    b. mix up the tested samples and clean samples, and record the entropy.
    c. detection samples whose entropy exceeds the thresholds as poisoned.


'''
import random
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader, Subset
from .base import Base

class STRIP(Base):
    def __init__(
        self,
        model: nn.Module,
        inspection_set: Dataset,
        clean_set: Dataset,
        num_classes: int = 10,
        strip_alpha: float = 1.0,
        N: int = 100,
        defense_fpr: float = 0.1,
        device: str | torch.device = 'cpu',

    ):
        super().__init__()

        self.model = model
        self.inspection_set = inspection_set
        self.clean_set = clean_set

        self.num_classes = num_classes
        self.strip_alpha = strip_alpha
        self.N = N
        self.defense_fpr = defense_fpr
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        # self._features_cache = None

    # def _register_hook(self, model: nn.Module):
    #     last_linear_layer = None
    #     for module in reversed(list(model.modules())):
    #         if isinstance(module, torch.nn.Linear):
    #             last_linear_layer = module
    #             break
        
    #     if last_linear_layer is None:
    #         raise ValueError("No FC layer founded in model")
        
    #     def hook(module, input, output):
    #         self._features_cache = input[0]
        
    #     last_linear_layer.register_forward_hook(hook)
    #     print("Hook registered for the last linear layer")

    # def _get_features(self, model: nn.Module, data_loader: DataLoader):

    #     class_indices = []
    #     feats = []

    #     model.eval()
    #     with torch.no_grad():
    #         for batch_idx, (data, target) in enumerate(data_loader):
    #             data, target = data.to(self.device), target.to(self.device)
    #             _ = model(data)
    #             feats.append(self._features_cache.detach().cpu().numpy())
    #             class_indices.append(target.detach().cpu().numpy())
    #     # concat all features and class indices
    #     feats = np.concatenate(feats, axis=0)
    #     class_indices = np.concatenate(class_indices, axis=0)
    #     return list(feats), list(class_indices)
    
    @torch.no_grad()
    def entropy(self, input: Tensor):
        p = nn.Softmax(dim=1)(self.model(input)) + 1e-8
        return (-p * torch.log(p)).sum(dim=1)

    @torch.no_grad()
    def _superimpose(self, input_1: Tensor, input_2: Tensor, alpha: float = None):
        if alpha is None:
            alpha = self.strip_alpha

        return input_1 + alpha * input_2
    
    @torch.no_grad()
    def _check(self, input: Tensor, source_set: Dataset):
        entropy_list = []

        samples = list(range(len(source_set)))
        random.shuffle(samples)
        samples = samples[:self.N]

        
        for i in samples:
            x, _ = source_set[i]
            x = x.to(self.device)
            entropy = self.entropy(self._superimpose(input, x)).cpu().detach()
            entropy_list.append(entropy)

        return torch.stack(entropy_list).mean(0)

    # @torch.no_grad()            
    # def _check(self, input: Tensor, source_set: Dataset):
    #     idxs = list(range(len(source_set)))
    #     random.shuffle(idxs)
    #     idxs = idxs[:self.N]

    #     probs_sum = None
    #     for i in idxs:
    #         x, _ = source_set[i]
    #         x = x.to(self.device)
    #         logits = self.model(self._superimpose(input, x))
    #         p = torch.softmax(logits, dim=1)
    #         probs_sum = p if probs_sum is None else (probs_sum + p)

    #     p_bar = probs_sum / self.N
    #     entropy = (-p_bar * torch.log(p_bar + 1e-8)).sum(dim=1)  # [batch]
    #     return entropy
    
    def cleanse(self):
        # choose a decision with the clean set
        clean_encropy = []
        clean_set_loader = DataLoader(self.clean_set, batch_size=128, shuffle=False)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(clean_set_loader):
                data, target = data.to(self.device), target.to(self.device)
                entropies = self._check(data, self.clean_set)
                clean_encropy.append(entropies.cpu())
        clean_encropy = np.concatenate(clean_encropy, axis=0)
        clean_encropy = torch.FloatTensor(clean_encropy)

        clean_entropy, _ = clean_encropy.sort()
        threshold_low = float(clean_entropy[int(self.defense_fpr * len(clean_encropy))])
        threshold_high = np.inf

        # now cleanse the inspection set with the chosen boundary
        inspection_set_loader = DataLoader(self.inspection_set, batch_size=128, shuffle=False)
        all_entropies = []
        for _, (data, target) in enumerate(inspection_set_loader):
            data, target = data.to(self.device), target.to(self.device)
            entropies = self._check(data, self.clean_set)
            all_entropies.append(entropies.cpu())
        all_entropies = np.concatenate(all_entropies, axis=0)
        all_entropies = torch.FloatTensor(all_entropies)

        suspicious_indices = torch.logical_or(
            all_entropies < threshold_low, all_entropies > threshold_high
            ).nonzero().reshape(-1)
        
        return suspicious_indices
    def filtering(self, y_true):
        # get the suspicious indices
        suspicious_indices = self.cleanse()
        y_preds = np.zeros(len(self.inspection_set), dtype=int)
        y_preds[suspicious_indices] = 1

        tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
        print("TPR: {:.2f}%".format(tp / (tp + fn) * 100))
        print("FPR: {:.2f}%".format(fp / (tn + fp) * 100))


        return y_preds
    
    @torch.no_grad()
    def collect_entropies(self, dataset):
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        ent_list = []
        for data, _ in loader:
            data = data.to(self.device)
            ent = self._check(data, self.clean_set)  # [batch]
            ent_list.append(ent)
        ent_arr = torch.cat(ent_list, dim=0).cpu().numpy()
        return ent_arr  # shape: [num_samples]
