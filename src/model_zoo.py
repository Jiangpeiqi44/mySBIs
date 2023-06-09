'''EFNB0/B4,Xcep,Res50,ViT-B'''

'''ViT-B'''

"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict
from model_HFF.networks.xception import TransferModel
from vit_model import vit_base_patch16_224_in21k
from swin_model import swin_tiny_patch4_window7_224
from resnet import resnet50,resnet34
import torch
import torch.nn as nn



from EFN_model import EfficientNet

def select_model(type):
    assert type in ['EFNB0','EFNB4','Xcep','Res50','Res34','ViT','S-ViT']
    if type =='ViT':
        model = vit_base_patch16_224_in21k(num_classes=2, has_logits=False)
        weight_pth = 'src/jx_vit_base_patch16_224_in21k-e5005f0a.pth'
        weights_dict = torch.load(weight_pth)
        # # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
        return model
    elif type=='Xcep':
        model = TransferModel('xception', dropout=0.5, inc=3)
        return model
    elif type=='EFNB0':
        model = EfficientNet.from_pretrained("efficientnet-b0",advprop=True,num_classes=2, weights_path='src/adv-efficientnet-b0-b64d5a18.pth')
        return model
    elif type=='EFNB4':
        model = EfficientNet.from_pretrained("efficientnet-b4",advprop=True,num_classes=2 ,weights_path='src/adv-efficientnet-b4-44fb3a87.pth')
        return model
    elif type=='Res50':
        model = resnet50(num_classes=2)
        weight_pth = 'src/resnet50-19c8e357.pth'
        weights_dict = torch.load(weight_pth)
        del weights_dict['fc.weight']
        del weights_dict['fc.bias']
        print(model.load_state_dict(weights_dict, strict=False))
        return model
    elif type=='Res34':
        model = resnet34(num_classes=2)
        weight_pth = 'src/resnet34-333f7ec4.pth'
        weights_dict = torch.load(weight_pth)
        del weights_dict['fc.weight']
        del weights_dict['fc.bias']
        print(model.load_state_dict(weights_dict, strict=False))
        return model
    if type =='S-ViT':
        model = swin_tiny_patch4_window7_224(num_classes=2)
        weight_pth = 'src/swin_tiny_patch4_window7_224.pth'
        weights_dict = torch.load(weight_pth)['model']
        # # 删除不需要的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
        return model
if __name__ == '__main__':
    select_model('EFNB4')
