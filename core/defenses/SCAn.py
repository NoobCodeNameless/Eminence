"""
This file is not the final version of SCAn[1]
All following codes are from the original SCAn paper
need to be modified for the current version of BackdoorBox

[1] Demon in the Variant: Statistical Analysis of DNNs for Robust Backdoor Contamination Detection
https://www.usenix.org/system/files/sec21-tang-di.pdf
"""

import numpy as np
import torch
from tqdm import tqdm
import os
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from copy import deepcopy
import torch.nn.functional as F
from torch import Tensor
from sklearn import metrics

from .base import Base

EPS = 1e-5

class SCAn(Base):
    def calc_final_score(self, lc_model=None):
        if lc_model is None:
            lc_model = self.lc_model
        sts = lc_model['sts']
        y = sts[:, 1]
        ai = self.calc_anomaly_index(y / np.max(y))
        return ai

    def calc_anomaly_index(self, a):
        ma = np.median(a)
        b = abs(a - ma)
        mm = np.median(b) * 1.4826  
        index = b / mm
        return index
     
    def build_global_model(self, reprs, labels, n_classes):

        N = reprs.shape[0]  # num_samples
        M = reprs.shape[1]  # len_features
        L = n_classes

        reprs = torch.FloatTensor(reprs).cuda()
        labels = torch.LongTensor(labels).cuda()
        mean_a = torch.mean(reprs, dim=0)
        X = reprs - mean_a

        cnt_L = torch.zeros(L).cuda()
        mean_f = torch.zeros(L,M).cuda()
        for k in range(L):
            idx = (labels == k)
            cnt_L[k] = torch.sum(idx)
            mean_f[k] = torch.mean(X[idx], dim=0)

        u = torch.zeros(N,M).cuda()
        e = torch.zeros([N, M]).cuda()
        for i in range(N):
            k = labels[i].item()
            u[i] = mean_f[k]  # class-mean
            e[i] = X[i] - u[i]  # sample-variantion
        Su = torch.cov(u.T)
        #np.cov(np.transpose(u))
        Se = torch.cov(e.T) # M x M
        #np.cov(np.transpose(e))

        # EM
        dist_Su = 1e5
        dist_Se = 1e5
        n_iters = 0
        while (dist_Su + dist_Se > EPS) and (n_iters < 100):

            n_iters += 1
            last_Su = Su
            last_Se = Se

            F = torch.pinverse(Se, rcond=1e-4)
            #torch.linalg.pinv(Se)
            #np.linalg.pinv(Se.cpu().numpy())
            #F = torch.FloatTensor(F).cuda()
            SuF = torch.matmul(Su, F)

            G_set = list()
            for k in range(L):
                #temp = (cnt_L[k] * Su + Se).cpu().numpy()
                #G = -np.linalg.pinv(temp)
                #G = torch.FloatTensor(G).cuda()
                G = -torch.pinverse(Su + Se/cnt_L[k], rcond=1e-4) / cnt_L[k]
                #-torch.linalg.pinv(Su + Se/cnt_L[k]) / cnt_L[k]
                G = torch.matmul(G, SuF)
                G_set.append(G)

            u_m = torch.zeros([L, M]).cuda()
            e = torch.zeros([N, M]).cuda()
            u = torch.zeros([N, M]).cuda()

            for i in range(N):
                vec = X[i:i+1]
                k = labels[i].item()
                G = G_set[k]
                dd = torch.matmul(torch.matmul(Se, G), vec.T).squeeze()
                u_m[k] = u_m[k] - dd


            for i in range(N):
                vec = X[i]
                k = labels[i].item()
                e[i] = vec - u_m[k]
                u[i] = u_m[k]

            # max-step
            Su = torch.cov(u.T)
            #np.cov(np.transpose(u))
            Se = torch.cov(e.T)
            #np.cov(np.transpose(e))

            dif_Su = Su - last_Su
            dif_Se = Se - last_Se

            dist_Su = torch.linalg.norm(dif_Su)
            #np.linalg.norm(dif_Su)
            dist_Se = torch.linalg.norm(dif_Se)
            #np.linalg.norm(dif_Se)
            # print(dist_Su,dist_Se)

            print('n_iter : %d, dist : %f' % (n_iters, dist_Su + dist_Se))

        gb_model = dict()
        gb_model['Su'] = Su.cpu().numpy()
        gb_model['Se'] = Se.cpu().numpy()
        gb_model['mean'] = mean_f.cpu().numpy()
        self.gb_model = gb_model
        return gb_model

    def build_local_model(self, reprs, labels, gb_model, n_classes):

        Su = gb_model['Su']
        Se = gb_model['Se']
        Su = torch.FloatTensor(Su).cuda()
        Se = torch.FloatTensor(Se).cuda()

        reprs = torch.FloatTensor(reprs).cuda()
        labels = torch.LongTensor(labels).cuda()

        #F = np.linalg.pinv(Se.cpu().numpy())
        #F = torch.FloatTensor(F).cuda()
        F = torch.pinverse(Se, rcond=1e-4)
        #torch.linalg.pinv(Se)
        #np.linalg.pinv(Se)
        N = reprs.shape[0]
        M = reprs.shape[1]
        L = n_classes

        mean_a = torch.mean(reprs, dim=0)
        #np.mean(reprs, axis=0)
        X = reprs - mean_a

        class_score = torch.zeros(L,3).cuda()
        #np.zeros([L, 3])
        u1 = torch.zeros(L,M).cuda()
        #np.zeros([L, M])
        u2 = torch.zeros(L,M).cuda()
        split_rst = list()

        for k in range(L):
            selected_idx = (labels == k)
            cX = X[selected_idx]
            subg, i_u1, i_u2 = self.find_split(cX, F)
            # print("subg",subg)
            i_sc = self.calc_test(cX, Su, Se, F, subg, i_u1, i_u2)
            split_rst.append(subg.cpu().numpy())
            u1[k] = i_u1.squeeze()
            u2[k] = i_u2.squeeze()
            class_score[k] = torch.tensor([k, i_sc, torch.sum(selected_idx).item()])

        lc_model = dict()
        lc_model['sts'] = class_score.cpu().numpy()
        lc_model['mu1'] = u1.cpu().numpy()
        lc_model['mu2'] = u2.cpu().numpy()
        lc_model['subg'] = split_rst

        self.lc_model = lc_model
        return lc_model

    def find_split(self, X, F):
        
        N = X.shape[0]
        M = X.shape[1]
        subg = torch.FloatTensor(np.random.rand(N)).cuda()

        if (N == 1):
            subg[0] = 0
            return (subg, X.clone(), X.clone())

        if torch.sum(subg >= 0.5) == 0:
            subg[0] = 1
        if torch.sum(subg < 0.5) == 0:
            subg[0] = 0
        last_z1 = -torch.ones(N).cuda()

        # EM
        steps = 0
        while (torch.linalg.norm(subg - last_z1) > EPS) and (torch.linalg.norm((1 - subg) - last_z1) > EPS) and (steps < 100):
            steps += 1
            last_z1 = subg.clone()

            # max-step
            # calc u1 and u2
            idx1 = (subg >= 0.5)
            idx2 = (subg < 0.5)
            if (torch.sum(idx1) == 0) or (torch.sum(idx2) == 0):
                break

            #u1 = torch.mean(X[idx1], dim=0, keepdim=True)
            #u2 = torch.mean(X[idx2], dim=0, keepdim=True)

            if torch.sum(idx1) == 1:
                u1 = X[idx1]
            else:
                u1 = torch.mean(X[idx1], dim=0, keepdim=True)

            if torch.sum(idx2) == 1:
                u2 = X[idx2]
            else:
                u2 = torch.mean(X[idx2], dim=0, keepdim=True)

            bias = torch.matmul(torch.matmul(u1, F), u1.T) - torch.matmul(torch.matmul(u2, F), u2.T)
            bias = bias.squeeze().item()

            e2 = u1 - u2
            for i in range(N):
                e1 = X[i:i+1]
                delta = torch.matmul(torch.matmul(e1, F), e2.T)
                delta = delta.squeeze().item()
                if bias - 2 * delta < 0:
                    subg[i] = 1
                else:
                    subg[i] = 0

        return (subg, u1, u2)

    """
    def calc_test(self, X, Su, Se, F, subg, u1, u2):

        N = X.shape[0]
        M = X.shape[1]

        temp = (N * Su + Se).cpu().numpy()
        G = -np.linalg.pinv(temp)
        G = torch.FloatTensor(G).cuda()
        #G = -torch.pinverse(Su + Se/N, rcond=1e-4) / N

        mu = torch.zeros(1,M).cuda()
        Se_mul_G = torch.matmul(Se, G)

        for i in range(N):
            vec = X[i:i+1]
            dd = torch.matmul(Se_mul_G, vec.T).T
            mu = mu - dd

        b1 = torch.matmul(torch.matmul(mu, F), mu.T) - torch.matmul(torch.matmul(u1, F), u1.T)
        b2 = torch.matmul(torch.matmul(mu, F), mu.T) - torch.matmul(torch.matmul(u2, F), u2.T)
        n1 = torch.sum(subg >= 0.5)
        n2 = N - n1
        sc = n1 * b1 + n2 * b2

        for i in range(N):
            e1 = X[i:i+1]
            if subg[i] >= 0.5:
                e2 = mu - u1
            else:
                e2 = mu - u2
            sc -= 2 * torch.matmul(torch.matmul(e1, F), e2.T)

        sc = sc.squeeze().item() / N
        print(N, n1, n2, sc)
        return sc
    """

    def calc_test(self, X, Su, Se, F, subg, u1, u2):
        N = X.shape[0]
        M = X.shape[1]

        X = X.cpu().numpy()
        Su = Su.cpu().numpy()
        Se = Se.cpu().numpy()
        F = F.cpu().numpy()
        subg = subg.cpu().numpy()
        u1 = u1.cpu().numpy()
        u2 = u2.cpu().numpy()

        G = -np.linalg.pinv(N * Su + Se)
        mu = np.zeros([1, M])
        Se_mul_G = np.matmul(Se, G)
        for i in range(N):
            vec = X[i]
            dd = np.matmul(Se_mul_G, np.transpose(vec))
            mu = mu - dd

        b1 = np.matmul(np.matmul(mu, F), np.transpose(mu)) - np.matmul(np.matmul(u1, F), np.transpose(u1))
        b2 = np.matmul(np.matmul(mu, F), np.transpose(mu)) - np.matmul(np.matmul(u2, F), np.transpose(u2))
        n1 = np.sum(subg >= 0.5)
        n2 = N - n1
        sc = n1 * b1 + n2 * b2

        print(N, n1, n2, sc)

        for i in range(N):
            e1 = X[i]
            if subg[i] >= 0.5:
                e2 = mu - u1
            else:
                e2 = mu - u2
            sc -= 2 * np.matmul(np.matmul(e1, F), np.transpose(e2))

        return sc / N

    def __init__(
            self,
            model: nn.Module,
            valset: Dataset,
            num_classes: int,
            threshold: float = np.e,
            seed: int=666,
            deterministic: bool=False,
            device: str | torch.device = 'cpu'
        ):
        super().__init__(seed=seed, deterministic=deterministic)
        self.device = device
        
        # and register a hook to the last linear layer
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        self.features_cache = None

        self._register_hook()

        self.valset = valset
        self.num_classes = num_classes
        self.threshold = threshold
        
    
    def _register_hook(self):
        last_linear_layer = None
        for module in reversed(list(self.model.modules())):
            if isinstance(module, torch.nn.Linear):
                last_linear_layer = module
                break
        
        if last_linear_layer is None:
            raise ValueError("No FC layer founded in model")
        
        def hook(module, input, output):
            self.features_cache = input[0]
        
        last_linear_layer.register_forward_hook(hook)
        print("Hook registered for the last linear layer")

    def _get_features(self, data_loader):
        # hook the layer, register a hook to the last linear layer
        
        features = []
        class_indices = []
        
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(self.device)
                _ = self.model(data)
                feature = self.features_cache
                features.append(feature.cpu().numpy())
                class_indices.append(target.cpu().numpy())
                
        return np.concatenate(features), np.concatenate(class_indices)
    
    def detect(self, dataset, batch_size=128):

        inspection_split_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        clean_split_loader = DataLoader(self.valset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        inspection_feats, inspection_labels = self._get_features(inspection_split_loader)
        clean_feats, clean_labels = self._get_features(clean_split_loader)

        inspection_feats = np.array(inspection_feats)
        inspection_labels = np.array(inspection_labels)

        clean_feats = np.array(clean_feats)
        clean_labels = np.array(clean_labels)

        feats_all = np.concatenate([inspection_feats, clean_feats])
        labels_all = np.concatenate([inspection_labels, clean_labels])
        
        gb_model = self.build_global_model(clean_feats, clean_labels, self.num_classes)
        lc_model = self.build_local_model(feats_all, labels_all, gb_model, self.num_classes)

        score = self.calc_final_score(lc_model)

        suspicious_indices = []

        for target_class in range(self.num_classes):
            print('[class-%d] outlier_score = %f' % (target_class, score[target_class]) )

            if score[target_class] <= self.threshold: continue

            tar_label = (labels_all == target_class)
            all_label = np.arange(len(labels_all))
            tar = all_label[tar_label]

            cluster_0_indices = []
            cluster_1_indices = []

            cluster_0_clean = []
            cluster_1_clean = []

            for index, i in enumerate(lc_model['subg'][target_class]):
                if i == 1:
                    if tar[index] > len(inspection_feats):
                        cluster_1_clean.append(tar[index])
                    else:
                        cluster_1_indices.append(tar[index])
                else:
                    if tar[index] > len(inspection_feats):
                        cluster_0_clean.append(tar[index])
                    else:
                        cluster_0_indices.append(tar[index])

            if len(cluster_0_clean) < len(cluster_1_clean):
                suspicious_indices += cluster_0_indices
            else:
                suspicious_indices += cluster_1_indices

        # suspicious indices are the indices of the samples in the inspection set
        # so the detection result is the 

        return suspicious_indices

    def test(self, testset, poisoned_indices, batch_size=1024):
        """
        Compute the TP, FP, TN, FN, acc, precision, recall, f1_score
        """
        
        # poison indices are the indices of the samples in the testset that are poisoned
        # 50% poisoned, 50% clean

        suspicious_indices = self.detect(testset, batch_size)

        TP = len(set(suspicious_indices) & set(poisoned_indices)) * 1.0 / len(poisoned_indices)
        FP = len(set(suspicious_indices) - set(poisoned_indices)) * 1.0 / len(suspicious_indices)

        TN = 1.0 - FP
        FN = 1.0 - TP

        detection_accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        if precision + recall != 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 'nan'

        return {
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN,
            'detection_accuracy': detection_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
def get_features(data_loader, model):

    class_indices = []
    feats = []

    model.eval()
    with torch.no_grad():
        for i, (ins_data, ins_target) in enumerate(tqdm(data_loader)):
            ins_data = ins_data.cuda()
            _, x_features = model(ins_data, True)

            this_batch_size = len(ins_target)
            for bid in range(this_batch_size):
                feats.append(x_features[bid].cpu().numpy())
                class_indices.append(ins_target[bid].cpu().numpy())

    return feats, class_indices

def cleanser(inspection_set, clean_set, model, num_classes):

    kwargs = {'num_workers': 4, 'pin_memory': True}

    # main dataset we aim to cleanse
    inspection_split_loader = torch.utils.data.DataLoader(
        inspection_set, batch_size=128, shuffle=False, **kwargs)

    # a small clean batch for defensive purpose
    clean_set_loader = torch.utils.data.DataLoader(
        clean_set,
        batch_size=128, shuffle=True, **kwargs)

    feats_inspection, class_indices_inspection = get_features(inspection_split_loader, model)
    feats_clean, class_indices_clean = get_features(clean_set_loader, model)

    feats_inspection = np.array(feats_inspection)
    class_indices_inspection = np.array(class_indices_inspection)

    feats_clean = np.array(feats_clean)
    class_indices_clean = np.array(class_indices_clean)

    # from sklearn.decomposition import PCA
    # projector = PCA(n_components=128)
    # feats_inspection = projector.fit_transform(feats_inspection)
    # feats_clean = projector.fit_transform(feats_clean)


    scan = SCAn()

    # fit the clean distribution with the small clean split at hand
    gb_model = scan.build_global_model(feats_clean, class_indices_clean, num_classes)

    size_inspection_set = len(feats_inspection)

    feats_all = np.concatenate([feats_inspection, feats_clean])
    class_indices_all = np.concatenate([class_indices_inspection, class_indices_clean])

    # use the global model to divide samples
    lc_model = scan.build_local_model(feats_all, class_indices_all, gb_model, num_classes)

    # statistic test for the existence of "two clusters"
    score = scan.calc_final_score(lc_model)
    threshold = np.e

    suspicious_indices = []

    for target_class in range(num_classes):

        print('[class-%d] outlier_score = %f' % (target_class, score[target_class]) )

        if score[target_class] <= threshold: continue

        tar_label = (class_indices_all == target_class)
        all_label = np.arange(len(class_indices_all))
        tar = all_label[tar_label]

        cluster_0_indices = []
        cluster_1_indices = []

        cluster_0_clean = []
        cluster_1_clean = []

        for index, i in enumerate(lc_model['subg'][target_class]):
            if i == 1:
                if tar[index] > size_inspection_set:
                    cluster_1_clean.append(tar[index])
                else:
                    cluster_1_indices.append(tar[index])
            else:
                if tar[index] > size_inspection_set:
                    cluster_0_clean.append(tar[index])
                else:
                    cluster_0_indices.append(tar[index])


        # decide which cluster is the poison cluster, according to clean samples' distribution
        if len(cluster_0_clean) < len(cluster_1_clean): # if most clean samples are in cluster 1
            suspicious_indices += cluster_0_indices
        else:
            suspicious_indices += cluster_1_indices

    return suspicious_indices

