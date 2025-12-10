"""
This is the implement of Eminence.

Eminence: A Trigger-based Backdoor Attack with Enhanced Stealthiness
"""
import os
from PIL import Image
from typing import Dict, Any, Literal
import copy
from tqdm import tqdm

import torch
from torch import nn, Tensor
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomAffine, ColorJitter, GaussianBlur
from torchvision.datasets import DatasetFolder
from torchvision.utils import save_image
import numpy as np

from .base import *
from collections import defaultdict

class NoiseTrigger:
    """
    NoiseTrigger class for applying a specific trigger pattern to images.

    This class inherits from the Trigger base class and implements the 
    functionality to blend a watermark (trigger) to an input image based on 
    a given pattern and weight. The pattern can be a 2D or 3D tensor, 
    and the weight determines the influence of the pattern on the final 
    watermarked image.

    Attributes:
        pattern (Tensor): The trigger pattern to be applied to the image.
        weight (Tensor): The weight tensor that controls the blending of 
                         the pattern with the input image.

    Methods:
        __call__(img: Tensor): Applies the trigger to the input 
                                          image and returns the watermarked 
                                          image tensor.
    """
    def __init__(self, pattern: Tensor, weight: Tensor):
        
        super().__init__()
        
        if pattern.dim() == 2:
            pattern = pattern.unsqueeze(0)
        
        if pattern.dim() != 3:
            raise ValueError('pattern shape should be 2 or 3')
        
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
            
        if weight.dim() != 3:
            raise ValueError('weight shape should be 2 or 3')
        
        
        self.pattern = pattern
        self.weight = weight
        
        
        # handling float type with range [0.0, 1.0]
        # and restrict the range of the watermarked image to [0.0, 1.0]
        
        print(f"NoiseTrigger initialized with pattern shape: {pattern.shape}, weight shape: {weight.shape}")
    
    @classmethod
    def add_trigger(cls, img: Tensor, pattern: Tensor, weight: Tensor):
        # restrict the range of the perturbed image to [0.0, 1.0]
        return torch.clamp(weight * img + (1.0 - weight) * pattern, 0.0, 1.0)
    
    def __call__(self, img: Tensor):

        """
        Adding a watermark on a benign image and then blend them as a trigger
        
        Args:
            img (Tensor): Input image tensor with shape (H, W) or (C, H, W).
        Returns:
            Tensor: Watermarked image tensor.
        """
         
        if img.dim() == 2:
            # H x W
            img = img.unsqueeze(0)
            img = self.add_trigger(img, self.pattern, self.weight)
            img = img.squeeze()
        elif img.dim() == 3:
            # C x H x W
            img = self.add_trigger(img, self.pattern, self.weight)
        else:
            raise ValueError('Input image shape should be 2 or 3')    
        
        return img.to(torch.float32)

class ProTriggerOptimizer:
    """
    
    """
    def __init__(
        self, 
        dataset: Dataset,
        model: nn.Module, 
        trigger_info: Dict[str, Any],
        device: str | torch.device = 'cpu',
        train_scale: float = 0.1,
        **kwargs, # for compatibility
        ):
        
        self.dataset = dataset
        self.model = model
        # self.pattern = trigger_info['pattern']
        
        self.pattern = trigger_info['pattern']
        self.weight = trigger_info['weight']
        self.device = device
        
        self.train_scale = train_scale
        # distill the dataset to train scale
        self.dataset = Subset(self.dataset, indices=torch.randperm(len(self.dataset))[:int(len(self.dataset) * self.train_scale)])
        
        
        self.model.to(self.device)
        self.model.eval()
        # hook toolkits
        # Register a hook to capture the input tensor of our need
        self.features_cache = None
        self._register_hook()

    def __call__(self, steps: int, lr: float = 0.05):
        
         # initialize the trigger or use the trigger offered
        pattern = self.pattern
        # pattern = pattern.to(self.device)
        pattern.requires_grad = True

        weight = self.weight
        # weight = weight.to(self.device)
        weight.requires_grad = False

        # Optimize the trigger arguments in need
        optimizer = optim.Adam([pattern], lr=lr)
        # optimizer = optim.Adam([pattern_params], lr=lr)
        
        print(f"Starting trigger optimization for {steps} steps with learning rate {lr}")
        
        for epoch in range(steps):  # we optimize steps epochs
            for batch in tqdm(DataLoader(self.dataset, batch_size=128, shuffle=True, num_workers=4), desc=f"Epoch {epoch} optimizing trigger"):
                images, labels = batch

                poisoned_images = self.apply_trigger_batch(images, pattern, weight)
                
                # print(f'former labels: {labels}')
                # if dirty_label >= 0:
                #     labels = torch.full_like(labels, dirty_label)
                #     print(f'latter labels: {labels}')
                
                poisoned_images = poisoned_images.to(self.device)
                labels = labels.to(self.device)
                
                # Get the input tensor of last fc layer
                features, logits = self.get_features(poisoned_images)
                print(f'labels: {labels}')
                print(f'pred: {logits.argmax(dim=1)}')
                
                # Compute the loss
                # loss = self.custom_loss(features)
                # redesign my loss
                loss = self.custom_loss(features)

                print(f'loss: {loss}')
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
        
        print("Trigger optimization completed")

        trigger_pattern = pattern.detach().cpu().clone()
        trigger_pattern = trigger_pattern.detach().cpu().clone()
        
        trigger_weight = weight.detach().cpu().clone()

        print(f'trigger_pattern value: {trigger_pattern}')

        trigger = NoiseTrigger(
            pattern=trigger_pattern,
            weight=trigger_weight
        )
        return trigger
    
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
    
    def get_features(self, images: Tensor):
        # Get the input tensor of the last FC layer
        logits = self.model(images)
        return self.features_cache, logits
    
    def custom_loss(self, features: Tensor):
        anchor = features[:1]
        other = features[1:]
        
        # cosine ones
        # return torch.cosine_similarity(anchor, other, dim=1).mean()
        # L2 ones
        return torch.norm(anchor - other, p=2, dim=1).mean()
    
    @classmethod    
    def apply_trigger_batch(self, images: Tensor, pattern: Tensor, weight: Tensor):
        """
        Applying trigger on batch of images
        
        """

        if images.dim() == 3 or images.dim() == 4:  # (B, H, W) or (B, C, H, W)
            poisoned_images = torch.stack([NoiseTrigger.add_trigger(img, pattern, weight) for img in images])
            return poisoned_images
        else:
            raise ValueError('Input image shape should be (B, H, W) or (B, C, H, W)')



class ProBANE(DatasetFolder):
    """
    The difference between ProBANE and BANE is that:
        - ProBANE focuses on the trigger attach process for each image
        - BANE cares too much, leads to a bad code structure
        - So this design enables this class compatible with Eminence triggers run with Eminence
    """
    def __init__(
        self, 
        benign_dataset: DatasetFolder,
        poison_ratio: float,
        # only accept trigger as a argument,
        # the left part is nothing to do with this class
        trigger,
        label_mode: Literal['CLEAN', 'DIRTY', 'MIXED'] = 'DIRTY',
        target_label: int = 0,
        num_classes: int = 10, # cifar10 default
    ):
        super().__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None
        )
        
        self.poison_ratio = poison_ratio
        self.trigger = trigger
        self.label_mode = label_mode
        self.target_label = target_label
        self.num_classes = num_classes

        self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.append(self.trigger)
        
        if self.target_transform is not None:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        else:
            self.poisoned_target_transform = Compose([])
        
        poisoned_num = int(len(self.samples) * poison_ratio)
            
        match self.label_mode:
            case 'DIRTY':
                print(f'[+] Dirty label mode: random samples are poisoned with the label {self.target_label} appointed by attackers')
                if poisoned_num > len(self.samples):
                    print(f'[+] poisoned_num is greater than total_num, set poisoned_num to total_num: {len(self.samples)}')
                    poisoned_num = len(self.samples)
                
                self.poisoned_set = random.sample(range(len(self.samples)), k=poisoned_num)
                
                self.poisoned_target_transform.transforms.append(lambda _: self.target_label)
                
            case 'CLEAN':
                print(f'[+] Clean label mode: random samples with the target label {self.target_label} are poisoned by attackers')
                target_class_indices = self.find_k_class_indices(self.target_label)
                if poisoned_num > len(target_class_indices):
                    print(f'[+] poisoned_num is greater than the target class total_num, set poisoned_num to target class total_num: {len(target_class_indices)}')
                    poisoned_num = len(target_class_indices)
                    
                self.poisoned_set = random.sample(target_class_indices, k=poisoned_num)
    
    def select_indices_by_label(self, poison_num: int):
        """
        Select the samples from different classes in ``poison_num`` samples
        """
        indices_nums = poison_num
        labels_dict = {}
        poisoned_list = []
        
        for k in range(self.num_classes):
            k_indices = self.find_k_class_indices(k)
            labels_dict[k] = k_indices
            
        while indices_nums > 0:
            for k in range(self.num_classes):
                if indices_nums <= 0:
                    break
                if labels_dict[k]:  # if the class has indices
                    index = random.choice(labels_dict[k])
                    poisoned_list.append(index)
                    labels_dict[k].remove(index)
                    indices_nums -= 1
        
        return poisoned_list
            
    def find_k_class_indices(self, k_class: int):
        """
        Find the indices of the k-th class in the dataset
        
        Args:
            k_class (int): The k-th class
        """
        return [index for index, (_, target) in enumerate(self.samples) if target == k_class]
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
    
        if index in self.poisoned_set:
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        
        return sample, target
    
class Eminence(Base):
    def __init__(
        self,
        train_dataset: DatasetFolder,
        test_dataset: DatasetFolder,
        model: nn.Module,
        loss: nn.Module,
        # Eminence arguments
        poison_ratio: float,
        trigger_info: Dict[str, Any],
        
        # Optimize arguments
        label_mode: Literal['CLEAN', 'DIRTY'] = 'DIRTY',
        target_label: int = 0,
        
        optimize_model: nn.Module | None = None,
        optimize_dataset: Dataset | None = None,
        optimize_device: str | torch.device = 'cpu',
        
        train_scale: float = 0.3,
        train_steps: int = 10,
        lr: float = 0.05,
        
        pretrained_trigger: NoiseTrigger | bool | None = None,
        
        schedule: Dict[str, Any] | None = None,
        seed: int = 0,
        deterministic: bool = False,
    ):
        super().__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic,
        )
    
        self.poison_ratio = poison_ratio
        self.trigger_info = trigger_info
        
        self.label_mode = label_mode
        self.target_label = target_label
        self.lr = lr
        # train_dataset
        self.num_classes = len(train_dataset.classes)
        
        # assuming surrogate model
        if optimize_model is None:
            self.optimize_model = copy.deepcopy(self.model)
        else:
            self.optimize_model = optimize_model
        
        # assuming surrogate dataset
        if optimize_dataset is None:
            self.optimize_dataset = self.train_dataset
        else:
            self.optimize_dataset = optimize_dataset
        
        self.optimize_device = optimize_device
        
        self.train_scale = train_scale
        self.train_steps = train_steps

        # For express experiment's sake
        # Handle trigger initialization based on pretrained_trigger parameter
        if isinstance(pretrained_trigger, NoiseTrigger):
            # Use provided trigger object directly
            self.trigger = pretrained_trigger
        elif pretrained_trigger is True:
            # In this case, the trigger info passing via 'trigger_info' argument
            # Create the trigger from trigger_info
            self.trigger = NoiseTrigger(
                pattern=self.trigger_info['pattern'],
                weight=self.trigger_info['weight']
            )
        else:
            # Generate new trigger if pretrained_trigger is None or False
            print("Generating trigger...")
            self.trigger = self.trigger_generation()
        
        
        
        self.poisoned_train_dataset = ProBANE(
            benign_dataset=self.train_dataset,
            poison_ratio=self.poison_ratio,
            trigger=self.trigger,
            label_mode=self.label_mode,
            target_label=self.target_label,
           
        )
        
        self.poisoned_test_dataset = ProBANE(
            benign_dataset=self.test_dataset,
            poison_ratio=1.0, # always poison all the samples
            trigger=self.trigger,
            label_mode='DIRTY', # always dirty label in ASR calculation cuz even in clean label mode, all the samples are poisoned and changed into target label
            target_label=self.target_label,
           
        )
        
        print("Eminence+ initialization completed")
        
    def trigger_generation(self):
        """
        Generate the trigger for the attack
        """
        trigger_optimizer = ProTriggerOptimizer(
            dataset=self.optimize_dataset,
            model=self.optimize_model,
            trigger_info=self.trigger_info,
            device=self.optimize_device,
            train_scale=self.train_scale,
        )
        
        trigger = trigger_optimizer(
            steps=self.train_steps,
            lr=self.lr,
        )
        
        return trigger
    
    def compute_asr(self):
        """
        Compute the ASR of the attack
        """
        # trigger = self.trigger
        # target = self.target_label
        print(f'[+] target label: {self.target_label}')
        print(f'[+] label mode: {self.label_mode}')
        
        # all_poisoned_dataset = ProBANE(
        #     benign_dataset=self.test_dataset,
        #     poison_ratio=1.1, # overload the benign dataset
        #     trigger=trigger,
        #     label_mode='DIRTY', # always dirty label in ASR calculation cuz even in clean label mode, all the samples are poisoned and changed into target label
        #     target_label=target,
        # )
        
        # return self._test(all_poisoned_dataset, device=self.optimize_device)
    
        return self._test(self.poisoned_test_dataset, device=self.optimize_device)
    
    def save_trigger(self):
        # for save the trigger
        return copy.deepcopy(self.trigger)