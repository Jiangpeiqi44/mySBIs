import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from PIL import Image
# import sys
import random
import warnings
# from utils.scheduler import LinearDecayLR
# from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse
from utils.logs import log
from utils.funcs import load_json
from datetime import datetime
from tqdm import tqdm
# from vit_custom_model import Vit_hDRMLPv3_ImageNet as Net
from model_zoo import select_model as Net
from torch.cuda.amp import autocast as autocast, GradScaler
import math
import torchvision.transforms as transforms

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBAL_DEVICES'] ='0'
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
from model_HFF.networks.xception import TransferModel
# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp)/len(pred_idx)

def seed_torch(seed=1029):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

class CosineAnnealingLRWarmup(torch.optim.lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=1.0e-8, last_epoch=-1, verbose=False,
                 warmup_steps=2, warmup_start_lr=1.0e-5):
        super(CosineAnnealingLRWarmup, self).__init__(optimizer, T_max=T_max,
                                                      eta_min=eta_min,
                                                      last_epoch=last_epoch)
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        if warmup_steps > 0:
            self.base_warup_factors = [
                (base_lr/warmup_start_lr)**(1.0/self.warmup_steps)
                for base_lr in self.base_lrs
            ]

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if hasattr(self, 'warmup_steps'):
            if self.last_epoch < self.warmup_steps:
                return [self.warmup_start_lr*(warmup_factor**self.last_epoch)
                        for warmup_factor in self.base_warup_factors]
            else:
                return [self.eta_min + (base_lr - self.eta_min) *
                        (1 + math.cos(math.pi * (self.last_epoch -
                         self.warmup_steps) / (self.T_max - self.warmup_steps)))*0.5
                        for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                    for base_lr in self.base_lrs]


def main():

    seed = 42   # 默认 seed = 5
    seed_torch(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda')

    image_size = 224
    batch_size = 20
    data_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    	     std=[0.229, 0.224, 0.225])
    ])
    train_dataset = torchvision.datasets.ImageFolder(root='../Dataset/AIGC/0509/train/real/',transform=data_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True,
                                               drop_last=True
                                               )
    train_dataset2 = torchvision.datasets.ImageFolder(root='../Dataset/AIGC/0509/train/fake/',transform=data_transform)

    train_loader2 = torch.utils.data.DataLoader(train_dataset2,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True,
                                               drop_last=True
                                               )
    
    # model = Net()
    model = Net('Xcep')
    model = model.to('cuda')
    n_epoch = 1
    act1=[]
    act2=[]
    for epoch in range(n_epoch):
        seed_torch(seed + epoch)
        model.train(mode=False)
        seed_torch(seed)
        for step, data in enumerate(tqdm(train_loader)):
            img = data[0].to(device, non_blocking=True).float()
            target = data[1]
            with torch.no_grad():
                output = model.features(img)
            print(output.shape)
            act1.append(output.cpu().data.numpy())
        for step, data in enumerate(tqdm(train_loader2)):
            img = data[0].to(device, non_blocking=True).float()
            target = data[1]
            with torch.no_grad():
                output = model.features(img)
            
            act2.append(output.cpu().data.numpy())

if __name__ == '__main__':

    main()
