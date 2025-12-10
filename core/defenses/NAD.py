'''
This is the implement of NAD [1]. 
This code is developed based on its official codes. (https://github.com/bboylyg/NAD)

Reference:
[1] Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks. ICLR 2021.
'''


import os
import os.path as osp
from copy import deepcopy
import time
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Base

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, MNIST, DatasetFolder

from ..utils.accuracy import accuracy
from ..utils.log import Log
from ..utils import test


class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks via Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self, p):
		super(AT, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am


class NAD(Base):
    """Repair a model via Neural Attention Distillation (NAD).

    Args:
        model (nn.Module): Repaired model.
        loss (nn.Module): Loss for repaired model training.
        power (float): The hyper-parameter for the attention loss.
        trainset (Dataset): The training dataset.
        portion (float): The portion of the training dataset to be used for the repairing training.
        beta (list): The hyper-parameter for the attention loss.
        target_layers (list): The target layers for the attention loss. 
                              Note that the coefficient of the attention loss for one layer in target_layers
                              is the value in beta in the same index as the layer.
        device (str): The device to run the model. Default: 'cpu'.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.

    
    """
    def __init__(self,
                 model,
                 loss, 
                 power,
                 poisoned_trainset=None,
                 poisoned_testset=None,
                 clean_trainset=None,
                 clean_testset=None,
                 portion=0.05,
                 beta=[],
                 target_layers=[],
                 device='cuda:0',
                 seed=0,
                 deterministic=False):
        super(NAD, self).__init__(seed, deterministic)

        assert len(beta)==len(target_layers), 'The length of beta must equal to the length of target_layers!'
        self.model = model
        self.loss = loss
        self.power = power
        self.beta = beta
        self.target_layers = target_layers
        self.seed = seed

        self.poisoned_trainset = poisoned_trainset
        self.poisoned_testset = poisoned_testset
        self.clean_trainset = clean_trainset
        self.clean_testset = clean_testset
        self.portion = portion
        self.device = device
        
        
    def get_model(self):
        return self.model

    def adjust_tune_learning_rate(self, optimizer, epoch):
        if epoch in self.current_schedule['schedule']:
            self.current_schedule['tune_lr'] *= self.current_schedule['gamma']
            for param_group in optimizer.param_groups:
                param_group['tune_lr'] = self.current_schedule['tune_lr']

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch in self.current_schedule['schedule']:
            self.current_schedule['lr'] *= self.current_schedule['gamma']
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.current_schedule['lr']

    def _train(self, dataset, portion, schedule):
        if schedule is None:
            raise AttributeError("Reparing Training schedule is None, please check your schedule setting.")
        elif schedule is not None: 
            self.current_schedule = deepcopy(schedule)

        # Use GPU
        if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in self.current_schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = self.current_schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert self.current_schedule['GPU_num'] >0, 'GPU_num should be a positive integer'
            print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {self.current_schedule['GPU_num']} of them to train.")

            if self.current_schedule['GPU_num'] == 1:
                device = torch.device("cuda:0")
            else:
                gpus = list(range(self.current_schedule['GPU_num']))
                self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        # get a portion of the repairing training dataset
        print("===> Loading {:.1f}% of traing samples.".format(portion*100))
        idxs = np.random.permutation(len(dataset))[:int(portion*len(dataset))]
        dataset = torch.utils.data.Subset(dataset, idxs)

        train_loader = DataLoader(
            dataset,
            batch_size=self.current_schedule['batch_size'],
            shuffle=True,
            num_workers=self.current_schedule['num_workers'],
            drop_last=False,
            pin_memory=True,
            worker_init_fn=self._seed_worker
        )
        self.train_loader=train_loader

        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        # log and output:
        # 1. ouput loss and time
        # 2. test and output statistics
        # 3. save checkpoint

        # Finetune and get the teacher model
        teacher_model = deepcopy(self.model)
        teacher_model = teacher_model.to(device)
        teacher_model.train()

        t_optimizer = torch.optim.SGD(teacher_model.parameters(), lr=self.current_schedule['tune_lr'], momentum=self.current_schedule['momentum'], weight_decay=self.current_schedule['weight_decay'])

        iteration = 0
        last_time = time.time()

        msg = f"Total train samples: {len(dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['tune_lr']}\n"
        log(msg)

        for i in range(self.current_schedule['tune_epochs']):
            self.adjust_tune_learning_rate(t_optimizer, i)
            for batch_id, batch in enumerate(train_loader):
                batch_img = batch[0]
                batch_label = batch[1]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                t_optimizer.zero_grad()
                predict_digits = teacher_model(batch_img)
                loss = self.loss(predict_digits, batch_label)
                loss.backward()
                t_optimizer.step()

                iteration += 1

                if iteration % self.current_schedule['log_iteration_interval'] == 0:
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"FineTune Epoch:{i+1}/{self.current_schedule['tune_epochs']}, iteration:{batch_id + 1}/{len(dataset)//self.current_schedule['batch_size']}, lr: {self.current_schedule['tune_lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                    last_time = time.time()
                    log(msg)
        
        # Save the teacher model
        teacher_model.eval()
        teacher_model = teacher_model.cpu()
        ckpt_model_filename = "teacher_model.pth"
        ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
        torch.save(teacher_model.state_dict(), ckpt_model_path)
        teacher_model = teacher_model.to(device)
        teacher_model.train()


        # Perform NAD and get the repaired model
        for param in teacher_model.parameters():
            param.requires_grad = False
        self.model = self.model.to(device)
        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_schedule['lr'], momentum=self.current_schedule['momentum'], weight_decay=self.current_schedule['weight_decay'])

        iteration = 0
        last_time = time.time()

        msg = f"Total train samples: {len(dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
        log(msg)

        criterionAT = AT(self.power)

        for i in range(self.current_schedule['epochs']):
            self.adjust_learning_rate(optimizer, i)
            for batch_id, batch in enumerate(train_loader):
                batch_img = batch[0]
                batch_label = batch[1]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                optimizer.zero_grad()

                container = []
                def forward_hook(module, input, output):
                    container.append(output)
                
                hook_list = []
                for name, module in self.model._modules.items():
                    if name in self.target_layers:
                        hk = module.register_forward_hook(forward_hook)
                        hook_list.append(hk)

                for name, module in teacher_model._modules.items():
                    if name in self.target_layers:
                        hk = module.register_forward_hook(forward_hook)
                        hook_list.append(hk)

                output_s = self.model(batch_img)
                _ = teacher_model(batch_img)

                for hk in hook_list:
                    hk.remove()

                loss = self.loss(output_s, batch_label)
                for idx in range(len(self.beta)):
                    loss = loss + criterionAT(container[idx], container[idx+len(self.beta)]) * self.beta[idx]   

                loss.backward()
                optimizer.step()

                iteration += 1

                if iteration % self.current_schedule['log_iteration_interval'] == 0:
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{self.current_schedule['epochs']}, iteration:{batch_id + 1}/{len(dataset)//self.current_schedule['batch_size']}, lr: {self.current_schedule['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                    last_time = time.time()
                    log(msg)
            
            if (i + 1) % self.current_schedule['save_epoch_interval'] == 0:
                self.model.eval()
                self.model = self.model.cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(i+1) + ".pth"
                ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                torch.save(self.model.state_dict(), ckpt_model_path)
                self.model = self.model.to(device)
                self.model.train()

    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
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
    
    def test(self, dataset, schedule):
        """Test repaired curve model on dataset

        Args:
            dataset (types in support_list): Dataset.
            schedule (dict): Schedule for testing.
        """
        model = self.model
        test(model, dataset, schedule)

    def repair(self, dataset=None, portion=None, schedule=None):
        """Perform NAD defense method based on attacked models. 
        The repaired model will be stored in self.model
        
        Args:
            dataset (types in support_list): Dataset.
            portion (float): in range(0,1), proportion of training dataset.
            schedule (dict): Schedule for Training the curve model.
        """
        print("===> Start training repaired model...")
        
        if dataset is None:
            dataset = self.clean_trainset
        if portion is None:
            portion = self.portion
        if schedule is None:
            schedule = {
                'device': 'GPU',
                'CUDA_SELECTED_DEVICES': self.device[:-1], # the last character is the device number
                'GPU_num': 1,

                'batch_size': 64,
                'num_workers': 8,
                'batch_size': 64,
                'num_workers': 8,

                'lr': 0.01,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'gamma': 0.1,

                'tune_lr': 0.01,
                'tune_epochs': 10,

                'epochs': 20,
                'schedule':  [2, 4, 6, 8], 

                'log_iteration_interval': 20,
                'test_epoch_interval': 20,
                'save_epoch_interval': 20,

                'save_dir': 'experiments/NAD-defense',
                'experiment_name': f'NAD_defense_portion_{portion}'
            }
            # print(schedule)
        
        former_CA, former_ASR = self.eval(self.model, self.clean_testset, self.poisoned_testset, self.device)
        
        print('==========Before NAD repairing==========')
        print(f'CA: {former_CA}, ASR: {former_ASR}')
        
        self._train(dataset, portion, schedule)
        
        latter_CA, latter_ASR = self.eval(self.model, self.clean_testset, self.poisoned_testset, self.device)
        
        print('==========After NAD repairing==========')
        print(f'CA: {latter_CA}, ASR: {latter_ASR}')
        
        
        
        