import os

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, ToPILImage, Resize

import core
from upload_model import ExperimentModel

dataset = torchvision.datasets.DatasetFolder

TRAIN_DATASET_PATH = '/home/fz/.local/share/cifar10/train'
TEST_DATASET_PATH = '/home/fz/.local/share/cifar10/test'

GPU = '0'

# image file -> cv.imread -> numpy.ndarray (H x W x C) -> ToTensor -> torch.Tensor (C x H x W) -> RandomHorizontalFlip -> torch.Tensor -> network input
transform_train = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor(),
    RandomHorizontalFlip()
])

trainset = dataset(
    root=TRAIN_DATASET_PATH,
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None
    )

transform_test = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor()
])
testset = dataset(
    root=TEST_DATASET_PATH,
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None
    )

attacker_dataset = dataset(
    root='/home/fz/.local/share/cifar100/train/',
    loader=cv2.imread,
    extensions=('png', 'jpeg'),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None
)

benign_model = core.models.ResNet(34, num_classes=10)
benign_model.load_state_dict(torch.load('pretrained_models/cifar10/resnet34/benign_model.pt', map_location='cpu'))

benign_model = core.models.ResNet(18, num_classes=10)


print('-' * 100)
print('Benign model training completed.')
print('-' * 100)

from core.attacks.Eminence import Eminence

trigger_size = 32
trigger_weight = 0.05

# trigger for tensor after ToTensor, with object range [0.0, 1.0]
pattern = torch.zeros((3, 32, 32), dtype=torch.float32)
pattern[:, -trigger_size:, -trigger_size:] = 1.0
weight = torch.ones((3, 32, 32), dtype=torch.float32)
weight[:, -trigger_size:, -trigger_size:] = (1 - trigger_weight)

eminence = Eminence(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, 10),
    loss=nn.CrossEntropyLoss(),
    # poison_ratio=0.0005,
    poison_ratio=0.0001,
    trigger_info={
        'pattern': pattern,
        'weight': weight
    },
    label_mode='DIRTY',
    target_label=0,
    
    train_scale=0.3,
    
    optimize_model=benign_model,
    optimize_dataset=trainset,
    optimize_device=torch.device(f'cuda:{GPU}')
)

schedule = {
    'device': 'GPU',
    # 'CUDA_VISIBLE_DEVICES': '0',
    'CUDA_SELECTED_DEVICES': GPU,
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 1024,
    'num_workers': 16,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'Eminence'
}

eminence.train(schedule)
eminence.test(schedule)