from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import math

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class CustomEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.hproj = torch.nn.Sequential(
            *[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4, bias=False),
              norm_layer(embed_dim//4),  # ????????????BN??????????????????LN
              nn.GELU(),
              RegionLayer(embed_dim//4, (8,8)),
              nn.Conv2d(embed_dim//4, embed_dim//4,
                        kernel_size=2, stride=2, bias=False),
              norm_layer(embed_dim//4),
              nn.GELU(),
              RegionLayer(embed_dim//4, (4,4)),
              nn.Conv2d(embed_dim//4, embed_dim,
                        kernel_size=2, stride=2, bias=False),
              norm_layer(embed_dim),
              ])
        
    def forward(self, x):
        x = self.hproj(x).flatten(2).transpose(1, 2)
        return x

class hDRMLPv2Embed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.hproj = torch.nn.Sequential(
            *[
              nn.Conv2d(in_chans, embed_dim//16, kernel_size=3, stride=1, padding=1, bias=False), # [768,224,224] -> [768//16,224,224]
              norm_layer(embed_dim//16),  # ????????????BN??????????????????LN
              nn.GELU(),
              
              RegionLayerDW(embed_dim//16, embed_dim//16, (7,7)), #[768//16,224,224] -> [768//16,224,224]
              nn.GELU(),
              
              nn.Conv2d(embed_dim//16, embed_dim//4, kernel_size=4, stride=4, bias=False), # [768//16,224,224] -> [768//4,56,56]
              norm_layer(embed_dim//4),  # ????????????BN??????????????????LN
              nn.GELU(),
              
              nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2, bias=False), # [768//4,56,56] -> [768//4,28,28]
              norm_layer(embed_dim//4),
              nn.GELU(),
              
              nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2, bias=False),  # [768//4,28,28] -> [768,14,14]
              norm_layer(embed_dim),
              ])
        
    def forward(self, x):
        x = self.hproj(x).flatten(2).transpose(1, 2)
        return x
        
class hDRMLPv3Embed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.hproj = torch.nn.Sequential(
            *[
              HierarchicalMultiScaleRegionLayer(in_chans, embed_dim//16),
              nn.Conv2d(embed_dim//16, embed_dim//4, kernel_size=4, stride=4, bias=False), # [768//16,224,224] -> [768//4,56,56]
              norm_layer(embed_dim//4),  # ????????????BN??????????????????LN
              nn.GELU(),
              
              nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2, bias=False), # [768//4,56,56] -> [768//4,28,28]
              norm_layer(embed_dim//4),
              nn.GELU(),
              
              nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2, bias=False),  # [768//4,28,28] -> [768,14,14]
              norm_layer(embed_dim),
              ])
        
    def forward(self, x):
        x = self.hproj(x).flatten(2).transpose(1, 2)
        return x

class hDRMLPv4Embed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.hproj = torch.nn.Sequential(
            *[
              nn.Conv2d(in_chans, embed_dim//16, kernel_size=3, stride=1, padding=1, bias=False), # [768,224,224] -> [768//16,224,224]
              norm_layer(embed_dim//16),  # ????????????BN??????????????????LN
              nn.GELU(),

              HierarchicalMultiScaleRegionLayerv2(embed_dim//16, embed_dim//16),
              # in -> in*(1+0.5+0.25)
              nn.Conv2d((int((1+0.5+0.25)*embed_dim//16)), embed_dim//4, kernel_size=4, stride=4, bias=False), # [768//16,224,224] -> [768//4,56,56]
              norm_layer(embed_dim//4),  # ????????????BN??????????????????LN
              nn.GELU(),
              
              nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2, bias=False), # [768//4,56,56] -> [768//4,28,28]
              norm_layer(embed_dim//4),
              nn.GELU(),
              
              nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2, bias=False),  # [768//4,28,28] -> [768,14,14]
              norm_layer(embed_dim),
              ])
        
    def forward(self, x):
        x = self.hproj(x).flatten(2).transpose(1, 2)
        return x

class hDRMLPv5Embed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.hproj = torch.nn.Sequential(
            *[
              nn.Conv2d(in_chans, embed_dim//16, kernel_size=3, stride=1, padding=1, bias=True), # [768,224,224] -> [768//16,224,224]

              RegionLayerDWBNv2_old(embed_dim//16, embed_dim//16, (7,7)), #[768//16,224,224] -> [768//16,224,224]
              norm_layer(embed_dim//16),
              nn.GELU(),
              
              nn.Conv2d(embed_dim//16, embed_dim//4, kernel_size=4, stride=4, bias=False), # [768//16,224,224] -> [768//4,56,56]
              norm_layer(embed_dim//4),  # ????????????BN??????????????????LN
              nn.GELU(),
              
              nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2, bias=False), # [768//4,56,56] -> [768//4,28,28]
              norm_layer(embed_dim//4),
              nn.GELU(),
              
              nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2, bias=False),  # [768//4,28,28] -> [768,14,14]
              norm_layer(embed_dim),
              ])
        
    def forward(self, x):
        x = self.hproj(x).flatten(2).transpose(1, 2)
        return x

class hDRMLPv6Embed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.hproj = torch.nn.Sequential(
            *[
              HierarchicalMultiScaleRegionLayerv3(in_chans, embed_dim//16), #[768//16,224,224] -> [768//16,224,224]
              
              nn.Conv2d(embed_dim//16, embed_dim//4, kernel_size=4, stride=4, bias=False), # [768//16,224,224] -> [768//4,56,56]
              norm_layer(embed_dim//4),  # ????????????BN??????????????????LN
              nn.GELU(),
              
              nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2, bias=False), # [768//4,56,56] -> [768//4,28,28]
              norm_layer(embed_dim//4),
              nn.GELU(),
              
              nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2, bias=False),  # [768//4,28,28] -> [768,14,14]
              norm_layer(embed_dim),
              ])
        
    def forward(self, x):
        x = self.hproj(x).flatten(2).transpose(1, 2)
        return x

class hDRMLPv7Embed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.hproj = torch.nn.Sequential(
            *[
              HierarchicalMultiScaleRegionLayerv4(in_chans, embed_dim//16),
              
              nn.Conv2d(embed_dim//16, embed_dim//4, kernel_size=4, stride=4, bias=False), # [768//16,224,224] -> [768//4,56,56]
              norm_layer(embed_dim//4),  # ????????????BN??????????????????LN
              nn.GELU(),
              
              nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2, bias=False), # [768//4,56,56] -> [768//4,28,28]
              norm_layer(embed_dim//4),
              nn.GELU(),
              
              nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2, bias=False),  # [768//4,28,28] -> [768,14,14]
              norm_layer(embed_dim),
              ])
        
    def forward(self, x):
        x = self.hproj(x).flatten(2).transpose(1, 2)
        return x

class hDRMLPv8Embed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.gelu = nn.GELU()
        self.region1 = torch.nn.Sequential(
            *[
              nn.Conv2d(in_chans, embed_dim//32, kernel_size=5, stride=1, padding=2, bias=False),
              norm_layer(embed_dim//32),
              nn.GELU(),
              RegionLayerDW(embed_dim//32,embed_dim//32,(7,7)),
            ])
        self.region2 = torch.nn.Sequential(
            *[
              nn.Conv2d(embed_dim//32, embed_dim//16, kernel_size=3, stride=1, padding=1, bias=False),
              norm_layer(embed_dim//16),
              nn.GELU(),
              RegionLayerDW(embed_dim//16,embed_dim//16,(4,4)),
            ])
        self.hMLP_branch1 = torch.nn.Sequential(
            *[
              nn.Conv2d(in_chans, embed_dim//16, kernel_size=4, stride=4, bias=False), # [768//16,224,224] -> [768//4,56,56]
              norm_layer(embed_dim//16),  # ????????????BN??????????????????LN
              nn.GELU(),
              nn.Conv2d(embed_dim//16, embed_dim//16, kernel_size=2, stride=2, bias=False), # [768//4,56,56] -> [768//4,28,28]
              norm_layer(embed_dim//16),
              nn.GELU(),
              nn.Conv2d(embed_dim//16, embed_dim//3, kernel_size=2, stride=2, bias=False),  # [768//4,28,28] -> [768,14,14]
              norm_layer(embed_dim//3),
              ])
        self.hMLP_branch2 = torch.nn.Sequential(
            *[
              nn.Conv2d(embed_dim//32, embed_dim//16, kernel_size=4, stride=4, bias=False), # [768//4,56,56] -> [768//4,28,28]
              norm_layer(embed_dim//16),
              nn.GELU(),
              nn.Conv2d(embed_dim//16, embed_dim//3, kernel_size=4, stride=4, bias=False),  # [768//4,28,28] -> [768,14,14]
              norm_layer(embed_dim//3),
              ])
        self.hMLP_branch3 = torch.nn.Sequential(
            *[
              nn.Conv2d(embed_dim//16, embed_dim//3, kernel_size=16, stride=16, bias=False),  # [768//4,28,28] -> [768,14,14]
              norm_layer(embed_dim//3),
              ])
    def forward(self, x):
        region1 = self.gelu(self.region1(x))
        region2 = self.gelu(self.region2(region1))
        branch1 = self.hMLP_branch1(x)
        branch2 = self.hMLP_branch2(region1)
        branch3 = self.hMLP_branch3(region2)
        x = torch.cat((branch1, branch2, branch3), 1)
        x = x.flatten(2).transpose(1, 2)
        return x

class hMLP_stem(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Sequential(
            *[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4, bias=False),
              norm_layer(embed_dim//4),  # ????????????BN??????????????????LN
              nn.GELU(),
              nn.Conv2d(embed_dim//4, embed_dim//4,
                        kernel_size=2, stride=2, bias=False),
              norm_layer(embed_dim//4),
              nn.GELU(),
              nn.Conv2d(embed_dim//4, embed_dim,
                        kernel_size=2, stride=2, bias=False),
              norm_layer(embed_dim),
              ])

    def forward(self, x):
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        if sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = h_sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class RegionLayer(nn.Module):
    def __init__(self, in_channels, grid=(8, 8)):
        super(RegionLayer, self).__init__()

        self.in_channels = in_channels
        self.grid = grid

        self.region_layers = dict()

        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                module_name = 'region_conv_%d_%d' % (i, j)
                # # ????????????
                # self.region_layers[module_name] = nn.Sequential(
                #     nn.BatchNorm2d(self.in_channels),
                #     nn.ReLU(),
                #     nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                #               kernel_size=3, stride=1, padding=1)
                # )
                # # Conv + BN + ReLu
                self.region_layers[module_name] = nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                              kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(self.in_channels),
                    nn.GELU()
                )
                # #
                self.add_module(name=module_name,
                                module=self.region_layers[module_name])

    def forward(self, x):
        """
        :param x:   (b, c, h, w)
        :return:    (b, c, h, w)
        """

        batch_size, _, height, width = x.size()

        input_row_list = torch.split(
            x, split_size_or_sections=height//self.grid[0], dim=2)
        output_row_list = []

        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = torch.split(
                row, split_size_or_sections=width//self.grid[1], dim=3)
            output_grid_list_of_a_row = []

            for j, grid in enumerate(input_grid_list_of_a_row):
                module_name = 'region_conv_%d_%d' % (i, j)
                grid = self.region_layers[module_name](
                    grid.contiguous()) + grid
                output_grid_list_of_a_row.append(grid)

            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            output_row_list.append(output_row)

        output = torch.cat(output_row_list, dim=2)

        return output

# ????????????(????????????)
class RegionLayerSD(nn.Module):
    def __init__(self, in_channels, out_channels, grid=(8, 8)):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid = grid

        self.region_layers = dict()

        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                module_name = 'region_conv_%d_%d' % (i, j)
                # # ????????????
                self.region_layers[module_name] = nn.Sequential(
                    nn.BatchNorm2d(self.in_channels),
                    nn.GELU(),
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=3, stride=1, padding=1, bias=True)
                )
                # #
                self.add_module(name=module_name,
                                module=self.region_layers[module_name])

    def forward(self, x):
        """
        :param x:   (b, c, h, w)
        :return:    (b, c, h, w)
        """
        batch_size, _, height, width = x.size()

        input_row_list = torch.split(
            x, split_size_or_sections=height//self.grid[0], dim=2)
        output_row_list = []

        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = torch.split(
                row, split_size_or_sections=width//self.grid[1], dim=3)
            output_grid_list_of_a_row = []

            for j, grid in enumerate(input_grid_list_of_a_row):
                module_name = 'region_conv_%d_%d' % (i, j)
                grid = self.region_layers[module_name](grid.contiguous())  #?????????????????????+ grid
                output_grid_list_of_a_row.append(grid)

            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            output_row_list.append(output_row)

        output = torch.cat(output_row_list, dim=2)

        return output

class RegionLayerDW(nn.Module):
    def __init__(self, in_channels, out_channels, grid=(8, 8)):
        super(RegionLayerDW, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid = grid

        self.region_layers = dict()

        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                module_name = 'region_conv_%d_%d' % (i, j)
                # # Conv + BN + ReLu
                self.region_layers[module_name] = nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                              kernel_size=3, stride=1, padding=1, groups=self.in_channels, bias=False),
                    nn.BatchNorm2d(self.in_channels),
                    nn.GELU(),
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,kernel_size=1, stride=1 ,bias=False),
                    nn.BatchNorm2d(self.out_channels),
                )
                # #
                self.add_module(name=module_name,
                                module=self.region_layers[module_name])

    def forward(self, x):
        batch_size, _, height, width = x.size()

        input_row_list = torch.split(
            x, split_size_or_sections=height//self.grid[0], dim=2)
        output_row_list = []

        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = torch.split(
                row, split_size_or_sections=width//self.grid[1], dim=3)
            output_grid_list_of_a_row = []

            for j, grid in enumerate(input_grid_list_of_a_row):
                module_name = 'region_conv_%d_%d' % (i, j)
                grid = self.region_layers[module_name](grid.contiguous()) + grid
                output_grid_list_of_a_row.append(grid)

            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            output_row_list.append(output_row)

        output = torch.cat(output_row_list, dim=2)

        return output
    
class RegionLayerDWv2(nn.Module):
    # conv bn relu??????
    def __init__(self, in_channels, out_channels, grid=(8, 8)):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid = grid

        self.region_layers = dict()
        self.res_1x1conv = dict()
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                module_name = 'region_conv_%d_%d' % (i, j)
                res_1x1conv_module_name = module_name+'_1x1conv'
                # # Conv + BN + ReLu
                self.region_layers[module_name] = nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                              kernel_size=3, stride=1, padding=1, groups=self.in_channels, bias=False),
                    nn.BatchNorm2d(self.in_channels),
                    nn.GELU(),
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,kernel_size=1, stride=1 ,bias=False),
                    nn.BatchNorm2d(self.out_channels)
                )
                # #
                self.add_module(name=module_name,
                                module=self.region_layers[module_name])
                self.res_1x1conv[res_1x1conv_module_name] = nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,kernel_size=1, stride=1 ,bias=False) #???????????????Bias??????????????????BN
                )
                self.add_module(name=res_1x1conv_module_name,
                                module=self.res_1x1conv[res_1x1conv_module_name])


    def forward(self, x):
        batch_size, _, height, width = x.size()

        input_row_list = torch.split(
            x, split_size_or_sections=height//self.grid[0], dim=2)
        output_row_list = []

        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = torch.split(
                row, split_size_or_sections=width//self.grid[1], dim=3)
            output_grid_list_of_a_row = []

            for j, grid in enumerate(input_grid_list_of_a_row):
                module_name = 'region_conv_%d_%d' % (i, j)
                res_1x1conv_module_name = module_name+'_1x1conv'
                grid = self.region_layers[module_name](grid.contiguous()) + self.res_1x1conv[res_1x1conv_module_name](grid.contiguous())
                output_grid_list_of_a_row.append(grid)

            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            output_row_list.append(output_row)

        output = torch.cat(output_row_list, dim=2)

        return output

class RegionLayerDWBN(nn.Module):
    ''' 
    ?????????BN??????,????????????Conv + Region + ??????BN (Region????????????????????????????????????BN+Gelu)
    ??????????????????Region???????????????BN???????????????????????????BN????????????????????????
    '''
    def __init__(self, in_channels, out_channels, grid=(8, 8)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid = grid
        self.region_layers = dict()
        self.res_1x1conv = dict()
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                module_name = 'region_conv_%d_%d' % (i, j)
                res_1x1conv_module_name = module_name+'_1x1conv'
                self.region_layers[module_name] = nn.Sequential(
                    nn.BatchNorm2d(self.in_channels),
                    nn.GELU(),
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                              kernel_size=3, stride=1, padding=1, groups=self.in_channels, bias=False),
                    nn.BatchNorm2d(self.in_channels),
                    nn.GELU(),
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,kernel_size=1, stride=1 ,bias=True) #???????????????Bias??????????????????BN
                )
                
                self.add_module(name=module_name,
                                module=self.region_layers[module_name])
                self.res_1x1conv[res_1x1conv_module_name] = nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,kernel_size=1, stride=1 ,bias=True) #???????????????Bias??????????????????BN
                )
                self.add_module(name=res_1x1conv_module_name,
                                module=self.res_1x1conv[res_1x1conv_module_name])

    def forward(self, x):
        batch_size, _, height, width = x.size()

        input_row_list = torch.split(
            x, split_size_or_sections=height//self.grid[0], dim=2)
        output_row_list = []

        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = torch.split(
                row, split_size_or_sections=width//self.grid[1], dim=3)
            output_grid_list_of_a_row = []

            for j, grid in enumerate(input_grid_list_of_a_row):
                module_name = 'region_conv_%d_%d' % (i, j)
                res_1x1conv_module_name = module_name+'_1x1conv'
                grid = self.region_layers[module_name](grid.contiguous()) + self.res_1x1conv[res_1x1conv_module_name](grid.contiguous())  # ???????????????????????????
                output_grid_list_of_a_row.append(grid)

            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            output_row_list.append(output_row)

        output = torch.cat(output_row_list, dim=2)

        return output

class RegionLayerDWBNv2_old(nn.Module):
    ''' 
    ?????????BN??????,????????????Conv + Region + ??????BN (Region????????????????????????????????????BN+Gelu)
    ??????????????????Region???????????????BN???????????????????????????BN????????????????????????
    '''
    def __init__(self, in_channels, out_channels, grid=(8, 8)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid = grid
        self.region_layers = dict()
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                module_name = 'region_conv_%d_%d' % (i, j)
                self.region_layers[module_name] = nn.Sequential(
                    nn.BatchNorm2d(self.in_channels),
                    nn.GELU(),
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                              kernel_size=3, stride=1, padding=1, groups=self.in_channels, bias=False),
                    nn.BatchNorm2d(self.in_channels),
                    nn.GELU(),
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,kernel_size=1, stride=1 ,bias=True) #???????????????Bias??????????????????BN
                )
                self.add_module(name=module_name,
                                module=self.region_layers[module_name])
                
    def forward(self, x):
        batch_size, _, height, width = x.size()

        input_row_list = torch.split(
            x, split_size_or_sections=height//self.grid[0], dim=2)
        output_row_list = []

        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = torch.split(
                row, split_size_or_sections=width//self.grid[1], dim=3)
            output_grid_list_of_a_row = []

            for j, grid in enumerate(input_grid_list_of_a_row):
                module_name = 'region_conv_%d_%d' % (i, j)
                grid = self.region_layers[module_name](grid.contiguous()) + grid  
                output_grid_list_of_a_row.append(grid)

            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            output_row_list.append(output_row)

        output = torch.cat(output_row_list, dim=2)

        return output

class RegionLayerDWBNv2(nn.Module):
    ''' 
    ?????????????????????
    DW?????????????????????BN GeLU????????????bias
    ?????????BN??????,????????????Conv + Region + ??????BN (Region????????????????????????????????????BN+Gelu)
    ??????????????????Region???????????????BN???????????????????????????BN????????????????????????
    '''
    def __init__(self, in_channels, out_channels, grid=(8, 8)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid = grid
        self.region_layers = dict()
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                module_name = 'region_conv_%d_%d' % (i, j)
                self.region_layers[module_name] = nn.Sequential(
                    nn.BatchNorm2d(self.in_channels),
                    nn.GELU(),
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1, groups=self.in_channels, bias=False),
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, bias=False)
                )
                self.add_module(name=module_name,
                                module=self.region_layers[module_name])
                
    def forward(self, x):
        batch_size, _, height, width = x.size()

        input_row_list = torch.split(
            x, split_size_or_sections=height//self.grid[0], dim=2)
        output_row_list = []

        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = torch.split(
                row, split_size_or_sections=width//self.grid[1], dim=3)
            output_grid_list_of_a_row = []

            for j, grid in enumerate(input_grid_list_of_a_row):
                module_name = 'region_conv_%d_%d' % (i, j)
                grid = self.region_layers[module_name](grid.contiguous())   #???????????????????????????????????????????????? + grid  
                output_grid_list_of_a_row.append(grid)

            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            output_row_list.append(output_row)

        output = torch.cat(output_row_list, dim=2)

        return output
    
class HierarchicalMultiScaleRegionLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True) # [768,224,224] -> [768//16,224,224] ???????????????Bias????????????????????????BN
        # BN ??? Gelu???Region???
        # ????????????????????????Region Layer
        self.branch1 = RegionLayerDWBN(self.out_channels, self.out_channels//2, (8,8))
        self.branch2 = RegionLayerDWBN(self.out_channels//2, self.out_channels//4, (4,4))
        self.branch3 = RegionLayerDWBN(self.out_channels//4, self.out_channels//4, (2,2))
        self.norm_layer = nn.BatchNorm2d(self.out_channels) # Region?????????BN
        self.gelu = nn.GELU()


    def forward(self, x):
        x = self.first_conv(x)
        local_branch1 = self.branch1(x)
        local_branch2 = self.branch2(local_branch1)
        local_branch3 = self.branch3(local_branch2)
        local_out = torch.cat((local_branch1, local_branch2, local_branch3), 1)
        out = x + local_out
        out = self.norm_layer(out)
        out = self.gelu(out)
        
        return out

class HierarchicalMultiScaleRegionLayerv2(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # BN ??? Gelu???Region???
        # ????????????????????????Region Layer
        self.first_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True) # [768,224,224] -> [768//16,224,224] ???????????????Bias????????????????????????BN
        # BN ??? Gelu???Region???
        # ????????????????????????Region Layer
        self.branch1 = RegionLayerDWBN(self.out_channels, self.out_channels//2, (8,8))
        self.branch2 = RegionLayerDWBN(self.out_channels//2, self.out_channels//4, (4,4))
        self.branch3 = RegionLayerDWBN(self.out_channels//4, self.out_channels//4, (2,2))
        self.norm_layer = nn.BatchNorm2d(self.out_channels) # Region?????????BN
        self.gelu = nn.GELU()


    def forward(self, x):
        local_branch1 = self.branch1(x)
        local_branch2 = self.branch2(self.gelu(self.bn1(self.conv1(local_branch1))))
        local_branch3 = self.branch3(self.gelu(self.bn2(self.conv2(local_branch2))))
        local_out = torch.cat((local_branch1, local_branch2, local_branch3), 1)
        out = self.conv_res(x) + local_out
        out = self.norm_layer(out)
        out = self.gelu(out)
        
        return out

class HierarchicalMultiScaleRegionLayerv3(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # BN ??? Gelu???Region???
        # ????????????????????????Region Layer,????????????????????????????????????
        self.first_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,kernel_size=5, stride=1, padding=2, bias=True)
        self.branch1 = RegionLayerDWBNv2(self.out_channels, self.out_channels//2, (8,8))
        self.branch2 = RegionLayerDWBNv2(self.out_channels//2, self.out_channels//4, (4,4))
        self.branch3 = RegionLayerDWBNv2(self.out_channels//4, self.out_channels//4, (2,2))
        
        self.norm_layer = nn.BatchNorm2d(self.out_channels) # Region?????????BN
        self.gelu = nn.GELU()


    def forward(self, x):
        x = self.first_conv(x)
        local_branch1 = self.branch1(x)
        local_branch2 = self.branch2(local_branch1)
        local_branch3 = self.branch3(local_branch2)
        local_out = torch.cat((local_branch1, local_branch2, local_branch3), 1)
        out = x + local_out
        out = self.norm_layer(out)
        out = self.gelu(out)
        
        return out

class HierarchicalMultiScaleRegionLayerv4(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_conv = nn.Conv2d(self.in_channels, self.out_channels//2, kernel_size=3, stride=1, padding=1, bias=False) # [768,224,224] -> [768//16,224,224] ???????????????Bias????????????????????????BN
        self.norm_layer_first = nn.BatchNorm2d(self.out_channels//2) # Region?????????BN
        # BN ??? Gelu???Region???
        # ????????????????????????Region Layer
        self.branch1 = RegionLayerDWv2(self.out_channels//2, self.out_channels//4, (8,8))
        self.branch2 = RegionLayerDW(self.out_channels//4, self.out_channels//4, (7,7)) #??????????????? ??????v2????????????????????? 
        self.gelu = nn.GELU()


    def forward(self, x):
        x = self.first_conv(x)
        x = self.norm_layer_first(x)
        local_branch1 = self.branch1(self.gelu(x))
        local_branch2 = self.branch2(self.gelu(local_branch1))
        local_out = torch.cat((x, local_branch1, local_branch2), 1)
        out = self.gelu(local_out) # ??????????????????bn ????????????gelu
        return out


class LocalityFeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, stride, expand_ratio=4., act='hs+se', reduction=4,
                 wo_dp_conv=False, dp_first=False):
        """
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
                    hs+se: h_swish and SE module
                    hs+eca: h_swish and ECA module
                    hs+ecah: h_swish and ECA module. Compared with eca, h_sigmoid is used.
        :param reduction: reduction rate in SE module.
        :param wo_dp_conv: without depth-wise convolution.
        :param dp_first: place depth-wise convolution as the first layer.
        """
        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3

        layers = []
        # the first linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)])

        # the depth-wise convolution between the two linear layers
        if not wo_dp_conv:
            dp = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)

        if act.find('+') >= 0:
            attn = act.split('+')[1]
            if attn == 'se':
                layers.append(SELayer(hidden_dim, reduction=reduction))
            elif attn.find('eca') >= 0:
                layers.append(ECALayer(hidden_dim, sigmoid=attn == 'eca'))
            else:
                raise NotImplementedError('Activation type {} is not implemented'.format(act))

        # the second linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_dim)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.conv(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,   # ??????token???dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # for get local
        attn_map = attn
        attn = self.attn_drop(attn_map)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Block_local(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 norm_layer=nn.LayerNorm, 
                 act_layer=nn.GELU,
                 act='hs', 
                 wo_dp_conv=False, 
                 dp_first=False):
        super(Block_local, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        # locality conv
        # The MLP is replaced by the conv layers.
        self.conv = LocalityFeedForward(dim, dim, 1, mlp_ratio, act, dim//4, wo_dp_conv, dp_first) # dim//4
  
    def forward(self, x):
        batch_size, num_token, embed_dim = x.shape                                  # (B, 197, dim)
        patch_size = int(math.sqrt(num_token))

        x = x + self.drop_path(self.attn(self.norm1(x)))                            # (B, 197, dim)
        # Split the class token and the image token.
        cls_token, x = torch.split(x, [1, num_token - 1], dim=1)                    # (B, 1, dim), (B, 196, dim)
        # Reshape and update the image token.
        x = x.transpose(1, 2).view(batch_size, embed_dim, patch_size, patch_size)   # (B, dim, 14, 14)
        x = self.conv(x).flatten(2).transpose(1, 2)                                 # (B, 196, dim)
        # Concatenate the class token and the newly computed image token.
        x = torch.cat([cls_token, x], dim=1)
        return x

class VisionTransformer_custom(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, isEmbed=True):

        super(VisionTransformer_custom, self).__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(
            1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[
                         i],
                     norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.isEmbed = isEmbed
        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(
            self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(
                self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        if self.isEmbed:
            x = self.patch_embed(x)  # [B, 196, 768]
            # x = self.patch_embed_hMLP(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(
                x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            # ?????????????????????[B,196,768]?????????token
            return self.pre_logits(x[:, 0]), x[:, 1:]
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x, patch_token = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x, patch_token

class VisionTransformer_local(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, isEmbed=True, depth_local=3):

        super(VisionTransformer_local, self).__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(
            1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(
        *(
            [Block_local(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                     norm_layer=norm_layer, act_layer=act_layer) for i in range(0, depth_local)]
            + [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                     norm_layer=norm_layer, act_layer=act_layer) for i in range(depth_local, depth)]
        ))
        self.norm = norm_layer(embed_dim)
        self.isEmbed = isEmbed
        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(
            self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(
                self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        if self.isEmbed:
            x = self.patch_embed(x)  # [B, 196, 768]
            # x = self.patch_embed_hMLP(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(
                x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            # ?????????????????????[B,196,768]?????????token
            return self.pre_logits(x[:, 0]), x[:, 1:]
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x, patch_token = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x, patch_token

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
        
def vit_base_patch16_224_in21k_custom(num_classes: int = 21843, has_logits: bool = True, isEmbed: bool = True, drop_ratio: float = 0.,
                                           attn_drop_ratio: float = 0., drop_path_ratio: float = 0., keepEmbedWeight: bool = True):
    model = VisionTransformer_custom(img_size=224,
                                          patch_size=16,
                                          embed_dim=768,
                                          depth=12,
                                          num_heads=12,
                                          drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=drop_path_ratio,
                                          representation_size=768 if has_logits else None,
                                          num_classes=num_classes, isEmbed=isEmbed)
    if not keepEmbedWeight:
        del model.patch_embed
    weight_pth = 'src/jx_vit_base_patch16_224_in21k-e5005f0a.pth'
    weights_dict = torch.load(weight_pth)
    # # ????????????????????????
    del_keys = ['head.weight', 'head.bias'] if model.has_logits \
        else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    # # # for DEBUG
    # weight_pth = 'output/3090_pretrained/ViT-4/8_0.9926_val.tar'
    # weights_dict = torch.load(weight_pth)["model"]
    print(model.load_state_dict(weights_dict, strict=False))
    return model

def vit_base_patch16_224_in21k_local(num_classes: int = 21843, has_logits: bool = True, isEmbed: bool = True, drop_ratio: float = 0.,
                                           attn_drop_ratio: float = 0., drop_path_ratio: float = 0., keepEmbedWeight: bool = True):
    model = VisionTransformer_local(img_size=224,
                                          patch_size=16,
                                          embed_dim=768,
                                          depth=12,
                                          num_heads=12,
                                          drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=drop_path_ratio,
                                          representation_size=768 if has_logits else None,
                                          num_classes=num_classes, isEmbed=isEmbed)
    if not keepEmbedWeight:
        del model.patch_embed
    weight_pth = 'src/jx_vit_base_patch16_224_in21k-e5005f0a.pth'
    weights_dict = torch.load(weight_pth)
    # # ????????????????????????
    del_keys = ['head.weight', 'head.bias'] if model.has_logits else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    print(model.load_state_dict(weights_dict, strict=False))
    
    # locality_vit_weight_pth = 'src/localvit_t_se4.pth'
    # locality_vit_weights_dict = torch.load(locality_vit_weight_pth)
    
    return model

def hLDR_embed(weight_pth:str):
    model = CustomEmbed(img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d)
    if weight_pth is not None:
        weights_dict = torch.load(weight_pth)['model']
        # # ????????????????????????
        keys = list(weights_dict.keys())
        # print(keys)
        # model_keys = []
        # for name, param in model.named_parameters():
        #     model_keys.append(name)
        # print(model_keys)
        for k in keys:
            if 'hproj' not in k:
                del weights_dict[k]
            else:
                print('Load: ',k)
        print(model.load_state_dict(weights_dict, strict=False))
        
    return model
    
def hDRMLPv2_embed(weight_pth:str):
    model = hDRMLPv2Embed(img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d)
    if weight_pth is not None:
        weights_dict = torch.load(weight_pth)['model']
        # # ????????????????????????
        keys = list(weights_dict.keys())
        for k in keys:
            if 'hproj' not in k:
                del weights_dict[k]
            else:
                print('Load: ',k)
        print(model.load_state_dict(weights_dict, strict=False))
        
    return model

def hDRMLPv3_embed(weight_pth:str):
    model = hDRMLPv3Embed(img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d)
    if weight_pth is not None:
        weights_dict = torch.load(weight_pth)['model']
        # # ????????????????????????
        keys = list(weights_dict.keys())
        for k in keys:
            if 'hproj' not in k:
                del weights_dict[k]
            else:
                print('Load: ',k)
        print(model.load_state_dict(weights_dict, strict=False))
        
    return model

def hDRMLPv4_embed(weight_pth:str):
    model = hDRMLPv4Embed(img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d)
    if weight_pth is not None:
        weights_dict = torch.load(weight_pth)['model']
        # # ????????????????????????
        keys = list(weights_dict.keys())
        for k in keys:
            if 'hproj' not in k:
                del weights_dict[k]
            else:
                print('Load: ',k)
        print(model.load_state_dict(weights_dict, strict=False))
    return model

def hDRMLPv5_embed(weight_pth:str):
    model = hDRMLPv5Embed(img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d)
    if weight_pth is not None:
        weights_dict = torch.load(weight_pth)['model']
        # # ????????????????????????
        keys = list(weights_dict.keys())
        for k in keys:
            if 'hproj' not in k:
                del weights_dict[k]
            else:
                print('Load: ',k)
        print(model.load_state_dict(weights_dict, strict=False))
    return model

def hDRMLPv6_embed(weight_pth:str):
    model = hDRMLPv6Embed(img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d)
    if weight_pth is not None:
        weights_dict = torch.load(weight_pth)['model']
        # # ????????????????????????
        keys = list(weights_dict.keys())
        for k in keys:
            if 'hproj' not in k:
                del weights_dict[k]
            else:
                print('Load: ',k)
        print(model.load_state_dict(weights_dict, strict=False))
    return model

def hDRMLPv7_embed(weight_pth:str):
    model = hDRMLPv7Embed(img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d)
    if weight_pth is not None:
        weights_dict = torch.load(weight_pth)['model']
        # # ????????????????????????
        keys = list(weights_dict.keys())
        for k in keys:
            if 'hproj' not in k:
                del weights_dict[k]
            else:
                print('Load: ',k)
        print(model.load_state_dict(weights_dict, strict=False))
    return model

def hDRMLPv8_embed(weight_pth:str):
    model = hDRMLPv8Embed(img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d)
    if weight_pth is not None:
        weights_dict = torch.load(weight_pth)['model']
        # # ????????????????????????
        keys = list(weights_dict.keys())
        for k in keys:
            if 'hproj' not in k:
                del weights_dict[k]
            else:
                print('Load: ',k)
        print(model.load_state_dict(weights_dict, strict=False))
    return model
class Vit_hDRMLP(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=2, has_logits=False, isEmbed=False, keepEmbedWeight=False)
        self.hproj = torch.nn.Sequential(
            *[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4, bias=False),
              norm_layer(embed_dim//4),  # ????????????BN??????????????????LN
              nn.GELU(),
              RegionLayer(embed_dim//4, (8,8)),
              nn.Conv2d(embed_dim//4, embed_dim//4,
                        kernel_size=2, stride=2, bias=False),
              norm_layer(embed_dim//4),
              nn.GELU(),
              RegionLayer(embed_dim//4, (4,4)),
              nn.Conv2d(embed_dim//4, embed_dim,
                        kernel_size=2, stride=2, bias=False),
              norm_layer(embed_dim),
              ])
        

    def forward(self, x):
        x = self.hproj(x).flatten(2).transpose(1, 2)
        cls_token, patch_token = self.vit_model(x)
        return cls_token

    def test_time(self, x):
        x = self.hproj(x).flatten(2).transpose(1, 2)
        cls_token, _ = self.vit_model(x)
        return cls_token

class Vit_consis_hDRMLP(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=2, has_logits=False, isEmbed=False, keepEmbedWeight=False)
        self.hproj = torch.nn.Sequential(
            *[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4, bias=False),
              norm_layer(embed_dim//4),  # ????????????BN??????????????????LN
              nn.GELU(),
              RegionLayer(embed_dim//4, (8,8)),
              nn.Conv2d(embed_dim//4, embed_dim//4,
                        kernel_size=2, stride=2, bias=False),
              norm_layer(embed_dim//4),
              nn.GELU(),
              RegionLayer(embed_dim//4, (4,4)),
              nn.Conv2d(embed_dim//4, embed_dim,
                        kernel_size=2, stride=2, bias=False),
              norm_layer(embed_dim),
              ])
        # consis-1
        self.K = nn.Linear(768, 768)
        self.Q = nn.Linear(768, 768)
        self.scale = 768 ** -0.5
        # # consis-2
        # self.K = nn.Identity()
        # self.Q = nn.Identity()

    def forward(self, x):
        x = self.hproj(x).flatten(2).transpose(1, 2)
        cls_token, patch_token = self.vit_model(x)
        # # consis-1
        consis_map = (self.K(patch_token) @
                      self.Q(patch_token).transpose(-2, -1)) * self.scale
        # # consis-2 add norm
        # consis_map_norm = torch.norm(patch_token, p=2, dim=2, keepdim=True)
        # consis_map = 0.5 + 0.5*((self.K(patch_token) @ self.Q(patch_token).transpose(-2, -1)) / (consis_map_norm@consis_map_norm.transpose(-2, -1)))
        return cls_token, consis_map

    def test_time(self, x):
        x = self.hproj(x).flatten(2).transpose(1, 2)
        cls_token, _ = self.vit_model(x)
        return cls_token

class Vit_hDRMLP_ImageNet(nn.Module):
    def __init__(self,  in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=100, has_logits=False, isEmbed=False, keepEmbedWeight=False, drop_ratio=0.1) #, drop_path_ratio=0.05
        self.hproj = torch.nn.Sequential(
            *[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4, bias=False),
              norm_layer(embed_dim//4),  # ????????????BN??????????????????LN
              nn.GELU(),
              RegionLayer(embed_dim//4, (8,8)),
              # nn.Identity(),
              nn.Conv2d(embed_dim//4, embed_dim//4,
                        kernel_size=2, stride=2, bias=False),
              norm_layer(embed_dim//4),
              nn.GELU(),
              RegionLayer(embed_dim//4, (4,4)),
              # nn.Identity(),
              nn.Conv2d(embed_dim//4, embed_dim,
                        kernel_size=2, stride=2, bias=False),
              norm_layer(embed_dim),
              ])
       
    def forward(self, x):
        x = self.hproj(x).flatten(2).transpose(1, 2)
        cls_token, _ = self.vit_model(x)
        return cls_token

class Vit_local(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_local(
            num_classes=2, has_logits=False, isEmbed=False, keepEmbedWeight=False)
        self.custom_embed = hLDR_embed(weight_pth)
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        return cls_token
    def test_time(self, x):
        return self.forward(x)

class Vit_consis_local(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_local(
            num_classes=2, has_logits=False, isEmbed=False, keepEmbedWeight=False)
        self.custom_embed = hLDR_embed(weight_pth)
        # consis-1
        self.K = nn.Linear(768, 768)
        self.Q = nn.Linear(768, 768)
        self.scale = 768 ** -0.5
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        # # consis-1
        consis_map = (self.K(patch_token) @
                      self.Q(patch_token).transpose(-2, -1)) * self.scale
        # # consis-2 add norm
        # consis_map_norm = torch.norm(patch_token, p=2, dim=2, keepdim=True)
        # consis_map = 0.5 + 0.5*((self.K(patch_token) @ self.Q(patch_token).transpose(-2, -1)) / (consis_map_norm@consis_map_norm.transpose(-2, -1)))
        return cls_token, consis_map

    def test_time(self, x):
        x = self.custom_embed(x)
        cls_token, _ = self.vit_model(x)
        return cls_token

class Vit_local_ImageNet(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_local(
            num_classes=100, has_logits=False, isEmbed=False, keepEmbedWeight=False, drop_ratio=0.2) # drop_ratio=0.1
        self.custom_embed = hLDR_embed(weight_pth)
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        return cls_token

    def test_time(self, x):
        return self.forward(x)

class Vit_hDRMLPv2(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=2, has_logits=False, isEmbed=False, keepEmbedWeight=False)
        self.custom_embed = hDRMLPv2_embed(weight_pth)
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        return cls_token
    def test_time(self, x):
        return self.forward(x)
    
class Vit_hDRMLPv2_ImageNet(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=100, has_logits=False, isEmbed=False, keepEmbedWeight=False, drop_ratio=0.1)
        self.custom_embed = hDRMLPv2_embed(weight_pth)
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        return cls_token
    def test_time(self, x):
        return self.forward(x)

class Vit_consis_hDRMLPv2(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=2, has_logits=False, isEmbed=False, keepEmbedWeight=False)
        self.custom_embed = hDRMLPv2_embed(weight_pth)
        # consis-1
        self.K = nn.Linear(768, 768)
        self.Q = nn.Linear(768, 768)
        self.scale = 768 ** -0.5
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        # # consis-1
        consis_map = (self.K(patch_token) @
                      self.Q(patch_token).transpose(-2, -1)) * self.scale
        # # consis-2 add norm
        # consis_map_norm = torch.norm(patch_token, p=2, dim=2, keepdim=True)
        # consis_map = 0.5 + 0.5*((self.K(patch_token) @ self.Q(patch_token).transpose(-2, -1)) / (consis_map_norm@consis_map_norm.transpose(-2, -1)))
        return cls_token, consis_map

    def test_time(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        return cls_token

class Vit_consis_hDRMLPv2_2(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=2, has_logits=False, isEmbed=False, keepEmbedWeight=False)
        self.custom_embed = hDRMLPv2_embed(weight_pth)
        # consis-1
        self.K = nn.Linear(768, 768)
        self.Q = nn.Linear(768, 768)
        self.scale = 768 ** -0.5
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        # # consis-1
        consis_map = (self.K(patch_token) @
                      self.Q(patch_token).transpose(-2, -1)) * self.scale
        # # consis-2 add norm
        # consis_map_norm = torch.norm(patch_token, p=2, dim=2, keepdim=True)
        # consis_map = 0.5 + 0.5*((self.K(patch_token) @ self.Q(patch_token).transpose(-2, -1)) / (consis_map_norm@consis_map_norm.transpose(-2, -1)))
        return cls_token, consis_map

    def test_time(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        return cls_token


class Vit_hDRMLPv3_ImageNet(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=100, has_logits=False, isEmbed=False, keepEmbedWeight=False, drop_ratio=0.1)
        self.custom_embed = hDRMLPv3_embed(weight_pth)
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        return cls_token
    def test_time(self, x):
        return self.forward(x)

class Vit_consis_hDRMLPv3(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=2, has_logits=False, isEmbed=False, keepEmbedWeight=False)
        self.custom_embed = hDRMLPv3_embed(weight_pth)
        # consis-1
        self.K = nn.Linear(768, 768)
        self.Q = nn.Linear(768, 768)
        self.scale = 768 ** -0.5
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        # # consis-1
        consis_map = (self.K(patch_token) @
                      self.Q(patch_token).transpose(-2, -1)) * self.scale
        # # consis-2 add norm
        # consis_map_norm = torch.norm(patch_token, p=2, dim=2, keepdim=True)
        # consis_map = 0.5 + 0.5*((self.K(patch_token) @ self.Q(patch_token).transpose(-2, -1)) / (consis_map_norm@consis_map_norm.transpose(-2, -1)))
        return cls_token, consis_map

    def test_time(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        return cls_token
    
class Vit_hDRMLPv4_ImageNet(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=100, has_logits=False, isEmbed=False, keepEmbedWeight=False, drop_ratio=0.1)
        self.custom_embed = hDRMLPv4_embed(weight_pth)
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        return cls_token
    def test_time(self, x):
        return self.forward(x)

class Vit_consis_hDRMLPv4(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=2, has_logits=False, isEmbed=False, keepEmbedWeight=False)
        self.custom_embed = hDRMLPv4_embed(weight_pth)
        # consis-1
        self.K = nn.Linear(768, 768)
        self.Q = nn.Linear(768, 768)
        self.scale = 768 ** -0.5
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        # # consis-1
        consis_map = (self.K(patch_token) @
                      self.Q(patch_token).transpose(-2, -1)) * self.scale
        # # consis-2 add norm
        # consis_map_norm = torch.norm(patch_token, p=2, dim=2, keepdim=True)
        # consis_map = 0.5 + 0.5*((self.K(patch_token) @ self.Q(patch_token).transpose(-2, -1)) / (consis_map_norm@consis_map_norm.transpose(-2, -1)))
        return cls_token, consis_map

    def test_time(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        return cls_token

class Vit_hDRMLPv5_ImageNet(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=100, has_logits=False, isEmbed=False, keepEmbedWeight=False, drop_ratio=0.1)
        self.custom_embed = hDRMLPv5_embed(weight_pth)
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        return cls_token
    def test_time(self, x):
        return self.forward(x)

class Vit_consis_hDRMLPv5(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=2, has_logits=False, isEmbed=False, keepEmbedWeight=False)
        self.custom_embed = hDRMLPv5_embed(weight_pth)
        # consis-1
        self.K = nn.Linear(768, 768)
        self.Q = nn.Linear(768, 768)
        self.scale = 768 ** -0.5
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        # # consis-1
        consis_map = (self.K(patch_token) @
                      self.Q(patch_token).transpose(-2, -1)) * self.scale
        # # consis-2 add norm
        # consis_map_norm = torch.norm(patch_token, p=2, dim=2, keepdim=True)
        # consis_map = 0.5 + 0.5*((self.K(patch_token) @ self.Q(patch_token).transpose(-2, -1)) / (consis_map_norm@consis_map_norm.transpose(-2, -1)))
        return cls_token, consis_map

    def test_time(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        return cls_token

class Vit_hDRMLPv6_ImageNet(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=100, has_logits=False, isEmbed=False, keepEmbedWeight=False, drop_ratio=0.1)
        self.custom_embed = hDRMLPv6_embed(weight_pth)
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        return cls_token
    def test_time(self, x):
        return self.forward(x)

class Vit_consis_hDRMLPv6(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=2, has_logits=False, isEmbed=False, keepEmbedWeight=False)
        self.custom_embed = hDRMLPv6_embed(weight_pth)
        # consis-1
        self.K = nn.Linear(768, 768)
        self.Q = nn.Linear(768, 768)
        self.scale = 768 ** -0.5
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        # # consis-1
        consis_map = (self.K(patch_token) @
                      self.Q(patch_token).transpose(-2, -1)) * self.scale
        # # consis-2 add norm
        # consis_map_norm = torch.norm(patch_token, p=2, dim=2, keepdim=True)
        # consis_map = 0.5 + 0.5*((self.K(patch_token) @ self.Q(patch_token).transpose(-2, -1)) / (consis_map_norm@consis_map_norm.transpose(-2, -1)))
        return cls_token, consis_map

    def test_time(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        return cls_token 
    
class Vit_hDRMLPv7_ImageNet(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=100, has_logits=False, isEmbed=False, keepEmbedWeight=False, drop_ratio=0.1)
        self.custom_embed = hDRMLPv7_embed(weight_pth)
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        return cls_token
    def test_time(self, x):
        return self.forward(x)

class Vit_consis_hDRMLPv7(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=2, has_logits=False, isEmbed=False, keepEmbedWeight=False)
        self.custom_embed = hDRMLPv7_embed(weight_pth)
        # consis-1
        self.K = nn.Linear(768, 768)
        self.Q = nn.Linear(768, 768)
        self.scale = 768 ** -0.5
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        # # consis-1
        consis_map = (self.K(patch_token) @
                      self.Q(patch_token).transpose(-2, -1)) * self.scale
        # # consis-2 add norm
        # consis_map_norm = torch.norm(patch_token, p=2, dim=2, keepdim=True)
        # consis_map = 0.5 + 0.5*((self.K(patch_token) @ self.Q(patch_token).transpose(-2, -1)) / (consis_map_norm@consis_map_norm.transpose(-2, -1)))
        return cls_token, consis_map

    def test_time(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        return cls_token 

class Vit_hDRMLPv8_ImageNet(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=100, has_logits=False, isEmbed=False, keepEmbedWeight=False, drop_ratio=0.1)
        self.custom_embed = hDRMLPv8_embed(weight_pth)
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        return cls_token
    def test_time(self, x):
        return self.forward(x)

class Vit_consis_hDRMLPv8(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_custom(
            num_classes=2, has_logits=False, isEmbed=False, keepEmbedWeight=False)
        self.custom_embed = hDRMLPv8_embed(weight_pth)
        # consis-1
        self.K = nn.Linear(768, 768)
        self.Q = nn.Linear(768, 768)
        self.scale = 768 ** -0.5
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        # # consis-1
        consis_map = (self.K(patch_token) @
                      self.Q(patch_token).transpose(-2, -1)) * self.scale
        # # consis-2 add norm
        # consis_map_norm = torch.norm(patch_token, p=2, dim=2, keepdim=True)
        # consis_map = 0.5 + 0.5*((self.K(patch_token) @ self.Q(patch_token).transpose(-2, -1)) / (consis_map_norm@consis_map_norm.transpose(-2, -1)))
        return cls_token, consis_map

    def test_time(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        return cls_token 
    
if __name__ == '__main__':
    from torchinfo import summary
    model = Vit_consis_hDRMLPv6()
    # print(model.vit_model.blocks)
    # print(model)
    image_size = 224
    batch_size = 1
    input_s = (batch_size, 3, image_size, image_size)
    summary(model, input_s)
    # dummy = torch.rand(batch_size, 3, image_size, image_size)
    # cls_token, consis_map = model(dummy)
    # print(consis_map.shape)
    '''
    for name, para in model.named_parameters():
        # ???head, pre_logits??? ????????????????????????
        if "head" not in name and "pre_logits" not in name:
            para.requires_grad_(False)
        else:
            print("training {}".format(name))
    '''
