from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F

from ..utils import test

from .base import Base

class LayerHooker:
    """
    Extract feature from the bottom of the layer of backdoored model
    
    Args:
        poisoned_model: the backdoored model
        model_type: the type of the model, currently only support ResNet
    """
    def __init__(
        self,
        poisoned_model: nn.Module,
        model_type: Literal['ResNet'] = 'ResNet'
    ):
        self.poisoned_model = poisoned_model
        self.poisoned_model.eval()
        self.model_type = model_type
        
        if self.model_type == 'ResNet':
            self.layer_name = 'layer4'
        else:
            raise NotImplementedError(f'{self.model_type} is not implemented')
        
        self.features = None
        self.__build_hook()
        
    def __build_hook(self):
        for name, module in self.poisoned_model.named_children():
            if isinstance(module, torch.nn.Sequential) and name == self.layer_name:
                self.hook = module.register_forward_hook(self.hook_fn)
         
    def hook_fn(self, module: nn.Module, input: Tensor, output: Tensor):
        self.features = input[0]
        
    def __call__(self, x: Tensor):
        logits = self.poisoned_model(x)
        return logits, self.features

class Beatrix(Base):
    """
    Identify backdoor samples using feature correlation analysis
    
    Args:
        model: the backdoored model
        model_type: the type of the model, currently only support ResNet
        order_list: The list of orders for Gram matrix calculation.
        seed: Global random seed
        deterministic: Whether to set the deterministic flag
    """
    def __init__(
        self, 
        model: nn.Module, 
        model_type: Literal['ResNet'] = 'ResNet', 
        order_list: list[int] = np.arange(1, 9), 
        seed: int = 666, 
        deterministic: bool = False
    ):
        super().__init__(seed, deterministic)
        
        self.model = model
        self.model.cuda()
        self.model.eval()
        
        self.model_type = model_type
        self.order_list = order_list
        
        self.layer_hooker = LayerHooker(self.model, self.model_type)
    
    @classmethod
    def Gram_p_matrix(self, x: Tensor, p: int):
        """
        Calculate the Gram matrix of the feature map
        
        Args:
            x: the feature map
            p: the order of the Gram matrix
            
        Returns:
            The Gram matrix
        """

        out = x.detach()
        out = out ** p
        out = out.reshape(out.shape[0], out.shape[1], -1)
        out = torch.matmul(out, out.transpose(dim0=2, dim1=1))
        out = out.triu()
        out = out.sign() * torch.abs(out) ** (1 / p)
        out = out.reshape(out.shape[0], -1)
        return out
    
    def get_deviations(self, features, medians, mads):
        deviations = []
        for feat in features:
            dev = 0
            for p, P in enumerate(self.order_list):
                g_p = self.Gram_p_matrix(feat, P)
                dev += torch.sum(torch.abs(g_p-medians[p])/(mads[p]+1e-6), dim=1, keepdim=True)
            deviations.append(dev.cpu().detach().numpy())
        return np.concatenate(deviations, axis=0)
    
    def _test(self, dataset: Dataset):
        data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
        self.model.eval()
        
        features = []
        pred_correct_mask = []
        
        with torch.no_grad():
            for batch in data_loader:
                imgs = batch[0].cuda()
                labels = batch[1].cuda() 
                
                logits, feature = self.layer_hooker(imgs)
                original_pred = torch.argmax(logits, dim=1)
                mask = torch.eq(labels, original_pred)
                
                pred_correct_mask.append(mask)
                features.append(feature)
                
        features = torch.cat(features, dim=0)
        pred_correct_mask = torch.cat(pred_correct_mask, dim=0)
        
        return features[pred_correct_mask], pred_correct_mask
    
    def test(self, clean_dataset: Dataset, poisoned_dataset: Dataset):
        clean_features, clean_pred_correct_mask = self._test(clean_dataset)
        poisoned_features, poisoned_pred_correct_mask = self._test(poisoned_dataset)
        
        medians = []
        mads = []    
        for P in self.order_list:
            g_p = self.Gram_p_matrix(clean_features[:self.clean_data_perclass], P)
            median = g_p.median(dim=0, keepdim=True)[0]
            mad = torch.abs(g_p - median).median(dim=0, keepdim=True)[0]
            medians.append(median)
            mads.append(mad)
        
        clean_deviations = self.get_deviations(clean_features, medians, mads)
        poisoned_deviations = self.get_deviations(poisoned_features, medians, mads)
        
        # Calculate metrics
        y_true = np.concatenate([np.zeros_like(clean_deviations), np.ones_like(poisoned_deviations)])
        y_score = np.concatenate([clean_deviations, poisoned_deviations])
        
        threshold = np.median(clean_deviations) + 2*np.std(clean_deviations)
        y_pred = (y_score >= threshold)
        
        # Calculate and print metrics
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        f1 = metrics.f1_score(y_true, y_pred)

        print("TPR: {:.2f}".format(tp / (tp + fn) * 100))
        print("FPR: {:.2f}".format(fp / (tn + fp) * 100))
        print("AUC: {:.4f}".format(auc))
        print(f"F1 Score: {f1:.4f}")

        return clean_deviations, poisoned_deviations
    
    def _detect(self, inputs: Tensor):
        inputs = inputs.cuda()
        _, features = self.layer_hooker(inputs)
        
        medians = []
        mads = []
        
        for P in self.order_list:
            g_p = self.Gram_p_matrix(features, P)
            median = g_p.median(dim=0, keepdim=True)[0]
            mad = torch.abs(g_p - median).median(dim=0, keepdim=True)[0]
            medians.append(median)
            mads.append(mad)
            
        deviations = self.get_deviations([features], medians, mads)
        threshold = np.median(deviations) + 2*np.std(deviations)
                
        return torch.tensor(deviations >= threshold)
    
    def detect(self, dataset, batch_size=128):
        """
        Return the detection poisoned rate of the input dataset
        """
        
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            all_preds = []
                    
            for idx, batch in enumerate(data_loader):
                imgs = batch[0]
                preds = self._detect(imgs)
                
            all_preds.append(preds)
            all_preds = torch.cat(all_preds)

        detected_poison_rate = torch.mean(all_preds.float()).item()
        
        return detected_poison_rate