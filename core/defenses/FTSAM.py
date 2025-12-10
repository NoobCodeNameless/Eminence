# This file is the implementation of FTSAM defense.
# FTSAM: Enhancing Fine-Tuning Based Backdoor Defense with Sharpness-Aware Minimization [ICCV, 2023] (https://arxiv.org/abs/2304.11823)

# Basic structure:
# 1. load the backdoored attack data and backdoored test data
# 2. load the backdoored model
# 3. for each round sample a clean batch from given clean subset:
#   a. do weight perturb to maximize L constrained by rho
#   b. do outer minimization
# 4. test the result and get ASR, ACC, RC

import os
import random
from tqdm import tqdm
import contextlib

import torch
import torch.nn.functional as F
from torch.utils.data import Subset, Dataset, DataLoader
import torch.nn as nn
from torch.distributed import ReduceOp
from torch.nn.modules.batchnorm import _BatchNorm

from .base import Base

def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

class ProportionScheduler:
    def __init__(self, pytorch_lr_scheduler, max_lr, min_lr, max_value, min_value):
        """
        This scheduler outputs a value that evolves proportional to pytorch_lr_scheduler, e.g.
        (value - min_value) / (max_value - min_value) = (lr - min_lr) / (max_lr - min_lr)
        """
        self.t = 0    
        self.pytorch_lr_scheduler = pytorch_lr_scheduler
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_value = max_value
        self.min_value = min_value
        
        assert (max_lr > min_lr) or ((max_lr==min_lr) and (max_value==min_value)), "Current scheduler for `value` is scheduled to evolve proportionally to `lr`," \
        "e.g. `(lr - min_lr) / (max_lr - min_lr) = (value - min_value) / (max_value - min_value)`. Please check `max_lr >= min_lr` and `max_value >= min_value`;" \
        "if `max_lr==min_lr` hence `lr` is constant with step, please set 'max_value == min_value' so 'value' is constant with step."
    
        assert max_value >= min_value
        
        self.step() # take 1 step during initialization to get self._last_lr
    
    def lr(self):
        return self._last_lr[0]
                
    def step(self):
        self.t += 1
        if hasattr(self.pytorch_lr_scheduler, "_last_lr"):
            lr = self.pytorch_lr_scheduler._last_lr[0]
        else:
            lr = self.pytorch_lr_scheduler.optimizer.param_groups[0]['lr']
            
        if self.max_lr > self.min_lr:
            value = self.min_value + (self.max_value - self.min_value) * (lr - self.min_lr) / (self.max_lr - self.min_lr)
        else:
            value = self.max_value
        
        self._last_lr = [value]
        return value


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, sam_alpha, rho_scheduler, adaptive=False, perturb_eps=1e-12, grad_reduce='mean', **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.rho_scheduler = rho_scheduler
        self.perturb_eps = perturb_eps
        self.alpha = sam_alpha
        
        # initialize self.rho_t
        self.update_rho_t()
        
        # set up reduction for gradient across workers
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else: # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')
    
    def disable_running_stats(self, model):
        def _disable(module):
            if isinstance(module, _BatchNorm):
                module.backup_momentum = module.momentum
                module.momentum = 0

        model.apply(_disable)

    def enable_running_stats(self, model):
        def _enable(module):
            if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum

        model.apply(_enable)
    
    @torch.no_grad()
    def update_rho_t(self):
        self.rho_t = self.rho_scheduler.step()
        return self.rho_t

    @torch.no_grad()
    def perturb_weights(self, rho=0.0):
        grad_norm = self._grad_norm( weight_adaptive = self.adaptive )
        for group in self.param_groups:
            scale = rho / (grad_norm + self.perturb_eps)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_g"] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w
                
    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, alpha=0.0):
        # calculate inner product
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                inner_prod += torch.sum(
                    self.state[p]['old_g'] * p.grad.data
                )

        # get norm
        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by='old_g')

        # get cosine
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        # gradient decomposition
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                vertical = self.state[p]['old_g'] - cosine * old_grad_norm * p.grad.data / (new_grad_norm + self.perturb_eps)
                p.grad.data.add_( vertical, alpha=-alpha)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized(): # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        #shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                    torch.stack([
                        ( (torch.abs(p.data) if weight_adaptive else 1.0) *  p.grad).norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        else:
            norm = torch.norm(
                torch.stack([
                    ( (torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        
    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.
        # This function does not take any arguments, and the inputs and targets data
        # should be pre-set in the definition of partial-function

        def get_grad():
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def step(self, closure=None):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            outputs, loss_value = get_grad()

            # perturb weights
            self.perturb_weights(rho=self.rho_t)

            # disable running stats for second pass
            self.disable_running_stats(self.model)

            # get gradient at perturbed weights
            get_grad()

            # decompose and get new update direction
            self.gradient_decompose(self.alpha)

            # unperturb
            self.unperturb()
            
        # synchronize gradients across workers
        self._sync_grad()    

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        self.enable_running_stats(self.model)

        return outputs, loss_value


class FTSAM(Base):
    """
    Repair a backdoor model via Fine-Tuning Sharpness-Aware Minimization (FTSAM).
    
    Args:
        model (nn.Module): Backdoor model to be repaired.
        loss (nn.Module): Loss for repaired model training.
        poisoned_trainset (type in support list): Poisoned trainset.
        poisoned_testset (types in support_list): Poisoned testset.
        clean_trainset (types in support_list): Clean trainset.
        clean_testset (types in support_list): Clean testset.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """
    def __init__(
        self,
        model,
        poisoned_trainset,
        poisoned_testset,
        clean_trainset,
        clean_testset,
        num_classes,
        device: str | torch.device = 'cpu',
        seed: int = 666, 
        deterministic: bool = False,
        loss: nn.Module = nn.CrossEntropyLoss(),
        epochs: int = 100,
        ratio: float = 0.05
    ):
        super().__init__(seed, deterministic)
        
        self.model = model
        
        self.poisoned_trainset = poisoned_trainset
        self.poisoned_testset = poisoned_testset
        self.clean_trainset = clean_trainset
        self.clean_testset = clean_testset
        self.num_classes = num_classes
        
        self.device = device
        self.loss = loss
        self.epochs = epochs
        self.ratio = ratio
        
        self.base_optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.base_optimizer,
            T_max=100,
        )
        self.rho_scheduler = ProportionScheduler(
            pytorch_lr_scheduler=self.lr_scheduler,
            max_lr=0.01,
            min_lr=0.0,
            max_value=2.0,
            min_value=2.0
        )
        
        self.optimizer = SAM(
            params=self.model.parameters(),
            base_optimizer=self.base_optimizer,
            model=self.model,
            sam_alpha=0.0,
            rho_scheduler=self.rho_scheduler,
            adaptive=False,
            perturb_eps=1e-12,
            grad_reduce='mean'
        )
    
    @classmethod
    def index_choicer_by_class(cls, dataset: Dataset, ratio: float, num_classes: int):
        
        dataset_length = len(dataset)
        class_indice_dict = [ [] for _ in range(num_classes) ]
        
        # traverse the dataset, and put the index of each class into the class_indice_dict
        for i in range(dataset_length):
            class_indice_dict[dataset[i][1]].append(i)
        
        indices = []
        
        # randomly choose the index of each class
        for i in range(num_classes):
            random.shuffle(class_indice_dict[i])
            class_indice_dict[i] = class_indice_dict[i][:int(len(class_indice_dict[i]) * ratio)]
            indices.extend(class_indice_dict[i])
        
        return indices
    
    @classmethod
    def eval(cls, model: nn.Module, clean_testset: Dataset, poisoned_testset: Dataset, device: str, batch_size: int=128):
        
        test_model = model.to(device)
        test_model.eval()
        
        clean_loader = DataLoader(clean_testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        poisoned_loader = DataLoader(poisoned_testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        def model_output(model, data_loader, device):
            predict_digits = []
            labels = []
            
            with torch.no_grad():
                
                for batch in data_loader:
                    batch_img, batch_label = batch
                    batch_img = batch_img.to(device)
                    batch_label = batch_label.to(device)
                    
                    batch_img = model(batch_img)
                    
                    predict_digits.append(batch_img.cpu())
                    labels.append(batch_label.cpu())
            
            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            
            return predict_digits, labels
        
        # compute CA
        clean_pred_digits, clean_labels = model_output(test_model, clean_loader, device)
        CA = (clean_pred_digits.argmax(dim=1) == clean_labels).sum().item() / clean_labels.size(0)
        
        # Compute ASR 
        
        poisoned_pred_digits, poisoned_labels = model_output(test_model, poisoned_loader, device)
        ASR = (poisoned_pred_digits.argmax(dim=1) == poisoned_labels).sum().item() / poisoned_labels.size(0)
        
        return CA, ASR
        
        
    def train(self, dataset, model, optimizer: SAM, scheduler, batch_size=128) -> nn.Module:
        model.train()
        
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        def loss_fn(preds, targets):
            return smooth_crossentropy(preds, targets, smoothing=0.1).mean()

        for i in tqdm(range(self.epochs), desc=f"in training epochs"):
            for (img, target) in tqdm(data_loader, desc="traverse data loader"):
                img = img.to(self.device)
                target = target.to(self.device)
                
                optimizer.set_closure(loss_fn, img, target)
                preds, _ = optimizer.step()
                
                with torch.no_grad():
                    correct = torch.argmax(preds.data, 1) == target
                    correct = correct.sum()
                    scheduler.step()
                    optimizer.update_rho_t()

        return model
        
    def repair(self):
        
        former_CA, former_ASR = self.eval(self.model, self.clean_testset, self.poisoned_testset, self.device)
        
        print('==========Before FTSAM repairing==========')
        print(f'CA: {former_CA}, ASR: {former_ASR}')
        
        
        clean_set = Subset(self.clean_trainset, self.index_choicer_by_class(self.clean_trainset, self.ratio, self.num_classes))

        self.model = self.train(clean_set, self.model, self.optimizer, self.lr_scheduler)
        
        latter_CA, latter_ASR = self.eval(self.model, self.clean_testset, self.poisoned_testset, self.device)
            
        print('==========After FTSAM repairing==========')
        print(f'CA: {latter_CA}, ASR: {latter_ASR}')
            
    def get_model(self):
        return self.model
            
