import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torchvision.transforms as transforms
import math

class NLBlockND(nn.Module):
    def __init__(self, in_channels=256):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        self.in_channels = in_channels

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions

        # add BatchNorm layer after the last conv layer
        self.sig = nn.Sigmoid()

        # define theta and phi for all operations except gaussian
        self.theta = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.in_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels,
                             out_channels=self.in_channels, kernel_size=1)

    def forward(self, x, return_nl_map=False):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation

        theta_x = self.theta(x).view(batch_size, self.in_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = f / math.sqrt(self.in_channels)

        # contiguous here just allocates contiguous chunk of memory
        y = f_div_C.permute(0, 2, 1).contiguous()

        sig_y = self.sig(y)
        # sig_y = y
        final_y = sig_y.view(batch_size, *x.size()[2:], *x.size()[2:])

        if return_nl_map:
            return final_y, sig_y
        else:
            return final_y



class AngleSimpleLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors"""
    def __init__(self, in_features, out_features):
        super(AngleSimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return cos_theta.clamp(-1, 1)


class SRMPixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SRMPixelAttention, self).__init__()
        # self.srm = SRMConv2d_simple()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.pa = SpatialAttention()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_srm):
        # x_srm = self.srm(x)
        fea = self.conv(x_srm)        
        att_map = self.pa(fea)
        
        return att_map


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=2048*2, out_chan=2048, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
        self.ca = ChannelAttention(out_chan, ratio=16)
        self.init_weight()

    def forward(self, x, y):
        fuse_fea = self.convblk(torch.cat((x, y), dim=1))
        fuse_fea = fuse_fea + fuse_fea * self.ca(fuse_fea)
        return fuse_fea

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class Two_Stream_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        self.xception_srm = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)

        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.srm_conv1 = SRMConv2d_Separate(32, 32)
        self.srm_conv2 = SRMConv2d_Separate(64, 64)
        self.relu = nn.ReLU(inplace=True)

        self.att_map = None
        self.srm_sa = SRMPixelAttention(3)
        self.srm_sa_post = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dual_cma0 = DualCrossModalAttention(in_dim=728, ret_att=False)
        self.dual_cma1 = DualCrossModalAttention(in_dim=728, ret_att=False)
        self.linPrj = nn.Sequential(
            nn.Conv2d(728, 256, 1, 1, 0, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.nlb = NLBlockND(256)
        self.fusion = FeatureFusionModule()
        self.anglelinear = AngleSimpleLinear(2048, 2)

        self.att_dic = {}

    def features(self, x):
        srm = self.srm_conv0(x)

        x = self.xception_rgb.model.fea_part1_0(x)
        y = self.xception_srm.model.fea_part1_0(srm) \
            + self.srm_conv1(x)
        y = self.relu(y)

        x = self.xception_rgb.model.fea_part1_1(x)
        y = self.xception_srm.model.fea_part1_1(y) \
            + self.srm_conv2(x)
        y = self.relu(y)

        # srm guided spatial attention
        self.att_map = self.srm_sa(srm)
        x = x * self.att_map + x
        x = self.srm_sa_post(x)

        x = self.xception_rgb.model.fea_part2(x)
        y = self.xception_srm.model.fea_part2(y)

        x, y = self.dual_cma0(x, y)


        x = self.xception_rgb.model.fea_part3(x)        
        y = self.xception_srm.model.fea_part3(y)

        # ## fea_part3 输出728,16,16

        x, y = self.dual_cma1(x, y)
        
        # ## 经过DCMA交互，输出728,16,16  利用RGB域的x做PCL
        mask = self.linPrj(x)
        mask = self.nlb(mask)
        # ## 继续原HFF网络
        x = self.xception_rgb.model.fea_part4(x)
        y = self.xception_srm.model.fea_part4(y)

        x = self.xception_rgb.model.fea_part5(x)
        y = self.xception_srm.model.fea_part5(y)

        fea = self.fusion(x, y)
                

        return fea, mask

    def classifier(self, fea):
        out, fea = self.xception_rgb.classifier(fea)
        return out, fea

    def forward(self, x):
        '''
        x: original rgb
        
        Return:
        out: (B, 2) the output for loss computing
        fea: (B, 1024) the flattened features before the last FC
        att_map: srm spatial attention map
        '''
        fea, mask = self.features(x)
        _, fea = self.classifier(fea)
        out = self.anglelinear(fea)

        return out, mask
    
if __name__ == '__main__':

    from components.attention import ChannelAttention, SpatialAttention, DualCrossModalAttention
    from components.srm_conv import SRMConv2d_simple, SRMConv2d_Separate
    from networks.xception import TransferModel
    # t_list = [transforms.ToTensor()]
    # composed_transform = transforms.Compose(t_list)

    # img = cv2.imread('out.jpg')
    # img = cv2.resize(img, (256, 256))
    # image = composed_transform(img)
    # image = image.unsqueeze(0)

    model = Two_Stream_Net()
    batch_size = 2
    dummy = torch.rand((batch_size ,3,256,256))
    out = model(dummy)
    print(out)
    # print(model)

else:
    from model_HFF.components.attention import ChannelAttention, SpatialAttention, DualCrossModalAttention
    from model_HFF.components.srm_conv import SRMConv2d_simple, SRMConv2d_Separate
    from model_HFF.networks.xception import TransferModel