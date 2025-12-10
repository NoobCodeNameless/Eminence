import copy
import torch
import torch.nn as nn
import numpy as np
import random
import os
from tqdm import tqdm
from typing import Literal, Dict, Any
from torch import Tensor
from torchvision.transforms import Compose
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from .base import *
from ..models import ResNet18

class AddNoise:
    def __init__(self, noise: Tensor):
        """
        Noise torch.Tensor(3, 32, 32)
        """   
        self.noise = noise

    def __call__(self, img: Tensor):
        """
        Args:
            img: Tensor of shape (3, 32, 32)
        Returns:
            Poisoned image Tensor of shape (3, 32, 32)
        """ 
        return torch.clamp(img + self.noise, -1, 1)

class ConcatDataset(Dataset):
    def __init__(self, target_dataset: Dataset, outer_dataset: Dataset):
        self.target_dataset = target_dataset
        self.outer_dataset = outer_dataset
        
    def __len__(self):
        return len(self.target_dataset) + len(self.outer_dataset)
    
    def __getitem__(self, idx):
        if idx < len(self.outer_dataset):
            img = self.outer_dataset[idx][0]
            label = self.outer_dataset[idx][1]
        else:
            img = self.target_dataset[idx - len(self.outer_dataset)][0]
            label = len(self.outer_dataset.classes)

        return img, label
    
    
class NarcissusDatasetFolder(DatasetFolder):
    
    def __init__(
        self, 
        benign_dataset: DatasetFolder,
        y_target: int,
        poisoned_rate: float,
        noise: Tensor,
        label_mode: Literal['CLEAN', 'DIRTY'] = 'CLEAN'
        ):
        
        super(NarcissusDatasetFolder, self).__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None
        )
        
        self.poisoned_rate = poisoned_rate
        self.y_target = y_target
        self.label_mode = label_mode
        
        self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.append(AddNoise(noise))
        
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        
        if self.target_transform is not None:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        else:
            self.poisoned_target_transform = Compose([])
        
        match self.label_mode:
            case 'CLEAN':
                print(f'[+] Clean label mode: random samples with the target label {self.y_target} are poisoned by attackers')
                target_class_indices = self.find_k_class_indices(self.y_target) 
                self.poisoned_set = random.sample(target_class_indices, k=poisoned_num)

            case 'DIRTY':
                print(f'[+] Dirty label mode: random samples are poisoned with the label {self.y_target} appointed by attackers')
                if poisoned_num > len(self.samples):
                    print(f'[+] poisoned_num is greater than total_num, set poisoned_num to total_num: {len(self.samples)}')
                    poisoned_num = len(self.samples)
                    
                self.poisoned_set = random.sample(range(len(self.samples)), k=poisoned_num)
                self.poisoned_target_transform.transforms.append(lambda _: self.y_target)
                    
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


class Narcissus(Base):
    def __init__(
        self,
        train_dataset: DatasetFolder,
        test_dataset: DatasetFolder,
        model: nn.Module,
        loss: nn.Module,
        y_target: int,
        poisoned_rate: float,
        outer_dataset: DatasetFolder,
        #
        
        schedule: Dict[str, Any] | None = None,
        seed: int = 66,
        deterministic: bool = False,
        test_device: str | torch.device = 'cpu',
    ):
        super(Narcissus, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic
        )
        
        self.y_target = y_target
        self.poisoned_rate = poisoned_rate
        self.test_device = test_device
        
        # generate the noise for the poisoned dataset
        self.noise = self.narcissus_gen(train_dataset, test_dataset, outer_dataset, y_target)
        
        self.poisoned_train_dataset = NarcissusDatasetFolder(
            self.train_dataset,
            self.y_target,
            self.poisoned_rate,
            self.noise,
            label_mode='CLEAN'
        )
        
        self.poisoned_test_dataset = NarcissusDatasetFolder(
            self.test_dataset,
            self.y_target,
            1.0,
            self.noise,
            label_mode='DIRTY'
        )
        
        # end of Narcissus, all leave for Base class
    
    def compute_asr(self):
        """
        Compute the ASR of the poisoned test dataset
        """
        return self._test(self.poisoned_test_dataset, self.test_device)
        
    def narcissus_gen(
        self,
        target_train_dataset: DatasetFolder,
        target_test_dataset: DatasetFolder,
        outer_dataset: DatasetFolder,
        y_target: int,
        ) -> Tensor:
        """
        Generate the noise for the poisoned dataset
        """
        
        noise_size = 32
        
        #Radius of the L-inf ball
        l_inf_r = 16/255
        
        # number of samples
        surrogate_model = ResNet18(num_classes=201)
        generating_model = ResNet18(num_classes=201)
        
        surrogate_epochs = 200
         #Learning rate for poison-warm-up
        generating_lr_warmup = 0.1
        warmup_round = 5
        
        #Learning rate for trigger generating
        generating_lr_tri = 0.01      
        gen_round = 1000
        
        #Training batch size
        # i dont know why it is 350
        train_batch_size = 350
        
        transform_surrogate_train = Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),  
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        from PIL import Image
        
        ori_train_dataset = DatasetFolder(
            root=target_train_dataset.root,
            loader=lambda path: Image.open(path).convert("RGB"),
            extensions=target_train_dataset.extensions,
            transform=transform_train,
            target_transform=target_train_dataset.target_transform,
        )

        ori_test_dataset = DatasetFolder(
            root=target_test_dataset.root,
            loader=lambda path: Image.open(path).convert("RGB"),
            extensions=target_test_dataset.extensions,
            transform=transform_test,
            target_transform=target_test_dataset.target_transform,
        )
        
        outer_dataset = DatasetFolder(
            root=outer_dataset.root,
            loader=lambda path: Image.open(path).convert("RGB"),
            extensions=outer_dataset.extensions,
            transform=transform_surrogate_train,
            target_transform=outer_dataset.target_transform,
        )
        
        # Step 1: Extract target class samples from target_train_dataset
        all_targets = [label for _, label in ori_train_dataset.samples]
        target_indices = [i for i, label in enumerate(all_targets) if label == y_target]

        target_subset = Subset(ori_train_dataset, target_indices)

        # Step 2: Create surrogate training set (target subset + POOD outer dataset)
        concoct_train_dataset = ConcatDataset(target_subset, outer_dataset)
        surrogate_loader = DataLoader(concoct_train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)

        # Step 3: Train surrogate model
        device = self.test_device
        surrogate_model = surrogate_model.to(device)
        criterion = nn.CrossEntropyLoss()
        surrogate_opt = torch.optim.SGD(surrogate_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        surrogate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(surrogate_opt, T_max=surrogate_epochs)

        print(f'[+] Training surrogate model for {surrogate_epochs} epochs')
        for epoch in range(surrogate_epochs):
            
            print(f'Epoch {epoch + 1} / {surrogate_epochs}')
            surrogate_model.train()
            for images, labels in surrogate_loader:
                images, labels = images.to(device), labels.to(device)
                surrogate_opt.zero_grad()
                outputs = surrogate_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                surrogate_opt.step()
            surrogate_scheduler.step()

        # Step 4: Warm-up generating model from surrogate weights
        generating_model.load_state_dict(surrogate_model.state_dict())
        generating_model = generating_model.to(device).train()
        warmup_loader = DataLoader(target_subset, batch_size=train_batch_size, shuffle=True, num_workers=4)
        warmup_opt = torch.optim.RAdam(generating_model.parameters(), lr=generating_lr_warmup)

        print(f'[+] Warm-up generating model for {warmup_round} rounds')
        for _ in range(warmup_round):
            for images, labels in warmup_loader:
                images, labels = images.to(device), labels.to(device)
                warmup_opt.zero_grad()
                outputs = generating_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                warmup_opt.step()

        # Step 5: Freeze model and optimize trigger noise
        for param in generating_model.parameters():
            param.requires_grad = False

        noise = torch.zeros((3, noise_size, noise_size), device=device, requires_grad=True)
        noise_opt = torch.optim.RAdam([noise], lr=generating_lr_tri)
        trigger_loader = DataLoader(target_subset, batch_size=train_batch_size, shuffle=True, num_workers=4)

        for step in tqdm(range(gen_round), desc="Trigger Optimization"):
            for images, labels in trigger_loader:
                images, labels = images.to(device), labels.to(device)
                # apply the noise to the batch of images
                perturbed = torch.clamp(images + torch.clamp(noise, -l_inf_r * 2, l_inf_r * 2), -1, 1)
                outputs = generating_model(perturbed)
                loss = criterion(outputs, labels)
                noise_opt.zero_grad()
                loss.backward()
                noise_opt.step()

            # if noise.grad is not None and noise.grad.abs().sum() == 0:
            #     print("[!] Gradient vanished. Stopping early.")
            #     break

        final_noise = torch.clamp(noise.detach(), -l_inf_r * 2, l_inf_r * 2).cpu()
        
        print(f'[+] Final noise optimization complete')
        return final_noise