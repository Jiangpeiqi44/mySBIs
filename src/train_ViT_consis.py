import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import sys
import random
import warnings
# from utils.ibi_wavelet import SBI_Dataset
# from utils.bi_wavelet import SBI_Dataset
# from utils.sbi_default import SBI_Dataset
from utils.MixBI_Map import SBI_Dataset
from utils.scheduler import LinearDecayLR
from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse
from utils.logs import log
from utils.funcs import load_json
from datetime import datetime
from tqdm import tqdm
# from my_xcep_vit_model import Vit_consis as Net
from vit_custom_model import Vit_consis_hDRMLP as Net
# from torch.cuda.amp import autocast as autocast, GradScaler
import math

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBAL_DEVICES'] ='0'
def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp)/len(pred_idx)


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


def main(args):
    cfg = load_json(args.config)

    seed = 42   # 默认 seed = 5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda')

    image_size = cfg['image_size']
    batch_size = cfg['batch_size']
    train_dataset = SBI_Dataset(
        phase='train', image_size=image_size, n_frames=8)
    val_dataset = SBI_Dataset(phase='val', image_size=image_size, n_frames=8)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size//2,
                                               shuffle=True,
                                               collate_fn=train_dataset.collate_fn,
                                               num_workers=15,
                                               pin_memory=True,
                                               drop_last=True,
                                               worker_init_fn=train_dataset.worker_init_fn
                                               )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             collate_fn=val_dataset.collate_fn,
                                             num_workers=15,
                                             pin_memory=True,
                                             worker_init_fn=val_dataset.worker_init_fn
                                             )

    model = Net()

    model = model.to('cuda')
     ## add 载入已训练
    if args.weight_name is not None:
        cnn_sd = torch.load(args.weight_name)["model"]
        
        del_keys = ['vit_model.head.weight', 'vit_model.head.bias']
        for k in del_keys:
            del cnn_sd[k]
            
        print(model.load_state_dict(cnn_sd,strict=False))
        print('Load pretrained model...')
 
        # for name, para in model.named_parameters():
        #     # 除head, pre_logits外 其他权重全部冻结
        #     if "head" not in name and "pre_logits" not in name and 'region_conv' not in name:
        #         para.requires_grad_(False)
        #     else:
        #         print("training {}".format(name))
                
    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(pg, lr=1e-3, momentum=0.9, weight_decay=5E-5)
    optimizer = torch.optim.AdamW(
        pg, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.3)  # 6e-5  3e-5  2e-5 wd默认1e-2
    # optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=2e-5, betas=(0.9, 0.999))
    iter_loss = []
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    val_accs = []
    val_losses = []
    n_epoch = cfg['epoch']
    # lr_scheduler = LinearDecayLR(optimizer, n_epoch, int(n_epoch/4*3))
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=n_epoch, eta_min=1e-10, last_epoch=-1, verbose=True)
    lr_scheduler = CosineAnnealingLRWarmup(optimizer,
                                           T_max=n_epoch,
                                           eta_min=1.0e-8,
                                           last_epoch=-1,
                                           warmup_steps=3,
                                           warmup_start_lr=1.0e-5)
    last_loss = 99999
    # scaler = GradScaler()
    now = datetime.now()
    # window环境下
    save_path = 'output/{}_'.format(args.session_name)+now.strftime(os.path.splitext(
        os.path.basename(args.config))[0])+'_'+now.strftime("%m_%d_%H_%M_%S")+'/'
    os.mkdir(save_path)
    os.mkdir(save_path+'weights/')
    os.mkdir(save_path+'logs/')
    logger = log(path=save_path+"logs/", file="losses.logs")

    criterion = nn.CrossEntropyLoss()
    criterionMap = nn.BCEWithLogitsLoss() #nn.BCELoss()
    lbda = 2
    last_auc = 0
    last_val_auc = 0
    weight_dict = {}
    n_weight = 3
    # 添加针对loss最小的几组pth
    last_val_loss = 0
    weight_dict_loss = {}
    n_weight_loss = 2

    for epoch in range(n_epoch):
        np.random.seed(seed + epoch)
        train_loss = 0.
        train_acc = 0.
        model.train(mode=True)
        for step, data in enumerate(tqdm(train_loader)):
            img = data['img'].to(device, non_blocking=True).float()
            target = data['label'].to(device, non_blocking=True).long()
            target_map = data['map'].to(device, non_blocking=True).float()
            optimizer.zero_grad()
            output, map = model(img)
            loss_cls = criterion(output, target)
            loss_map = criterionMap(map, target_map)
            loss = loss_cls + lbda*loss_map
            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss_value = loss.item()
            iter_loss.append(loss_value)
            train_loss += loss_value
            acc = compute_accuray(F.log_softmax(output, dim=1), target)
            train_acc += acc
        lr_scheduler.step()
        print('lr: ', lr_scheduler.get_last_lr())
        train_losses.append(train_loss/len(train_loader))
        train_accs.append(train_acc/len(train_loader))

        log_text = "Epoch {}/{} | train loss: {:.4f}, train acc: {:.4f}, ".format(
            epoch+1,
            n_epoch,
            train_loss/len(train_loader),
            train_acc/len(train_loader),
        )

        model.train(mode=False)
        val_loss = 0.
        val_acc = 0.
        output_dict = []
        target_dict = []
        np.random.seed(seed)
        for step, data in enumerate(tqdm(val_loader)):
            img = data['img'].to(device, non_blocking=True).float()
            target = data['label'].to(device, non_blocking=True).long()
            target_map = data['map'].to(device, non_blocking=True).float()
            with torch.no_grad():
                output, map = model(img)
                loss = criterion(output, target)
            loss_value = loss.item()
            iter_loss.append(loss_value)
            val_loss += loss_value
            acc = compute_accuray(F.log_softmax(output, dim=1), target)
            val_acc += acc
            output_dict += output.softmax(1)[:, 1].cpu().data.numpy().tolist()
            target_dict += target.cpu().data.numpy().tolist()
        val_losses.append(val_loss/len(val_loader))
        val_accs.append(val_acc/len(val_loader))
        val_auc = roc_auc_score(target_dict, output_dict)
        log_text += "val loss: {:.4f}, val acc: {:.4f}, val auc: {:.4f}".format(
            val_loss/len(val_loader),
            val_acc/len(val_loader),
            val_auc
        )

        if len(weight_dict) < n_weight:
            save_model_path = os.path.join(
                save_path+'weights/', "{}_{:.4f}_val.tar".format(epoch+1, val_auc))
            weight_dict[save_model_path] = val_auc
            torch.save({
                "model": model.state_dict(),
                # "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, save_model_path)
            last_val_auc = min([weight_dict[k] for k in weight_dict])

        elif val_auc >= last_val_auc:
            save_model_path = os.path.join(
                save_path+'weights/', "{}_{:.4f}_val.tar".format(epoch+1, val_auc))
            for k in weight_dict:
                if weight_dict[k] == last_val_auc:
                    del weight_dict[k]
                    os.remove(k)
                    weight_dict[save_model_path] = val_auc
                    break
            torch.save({
                "model": model.state_dict(),
                # "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, save_model_path)
            last_val_auc = min([weight_dict[k] for k in weight_dict])

        # ## 针对loss最小添加筛选
        if len(weight_dict_loss) < n_weight_loss:
            save_model_path = os.path.join(
                save_path+'weights/', "{}_{:.4f}_MINloss.tar".format(epoch+1, val_loss/len(val_loader)))
            weight_dict_loss[save_model_path] = val_loss/len(val_loader)
            torch.save({
                "model": model.state_dict(),
                # "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, save_model_path)
            last_val_loss = max([weight_dict_loss[k]
                                for k in weight_dict_loss])

        elif val_loss/len(val_loader) <= last_val_loss:
            save_model_path = os.path.join(
                save_path+'weights/', "{}_{:.4f}_MINloss.tar".format(epoch+1, val_loss/len(val_loader)))
            for k in weight_dict_loss:
                if weight_dict_loss[k] == last_val_loss:
                    del weight_dict_loss[k]
                    os.remove(k)
                    weight_dict_loss[save_model_path] = val_loss / \
                        len(val_loader)
                    break
            torch.save({
                "model": model.state_dict(),
                # "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, save_model_path)
            last_val_loss = max([weight_dict_loss[k]
                                for k in weight_dict_loss])

        logger.info(log_text)
    save_model_path = os.path.join(
        save_path+'weights/', "{}_last.tar".format(epoch+1))
    torch.save({
        "model": model.state_dict(),
        # "optimizer": model.optimizer.state_dict(),
        "epoch": epoch
    }, save_model_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='config')
    parser.add_argument('-n', dest='session_name')
    parser.add_argument('-w', dest='weight_name')
    args = parser.parse_args()
    main(args)
