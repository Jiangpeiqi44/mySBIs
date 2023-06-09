from functools import partial
from collections import OrderedDict
from cv2 import transpose
import torch
import torch.nn as nn
import math
import numpy as np
'''这个是基于UIA修改的一致性监督,修复attn map的一些bug
   此外 放弃了双重Map监督的思路'''

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
              norm_layer(embed_dim//16),  # 这里采用BN，也可以采用LN
              nn.GELU(),
              
              RegionLayerDW(embed_dim//16, embed_dim//16, (7,7)), #[768//16,224,224] -> [768//16,224,224]
              nn.GELU(),
              
              nn.Conv2d(embed_dim//16, embed_dim//4, kernel_size=4, stride=4, bias=False), # [768//16,224,224] -> [768//4,56,56]
              norm_layer(embed_dim//4),  # 这里采用BN，也可以采用LN
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
        
class RegionLayerDW(nn.Module):
    def __init__(self, in_channels, out_channels, grid=(8, 8)):
        super().__init__()

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
    
class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
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

class Attention_qk_map(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super().__init__()
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
        attn_qk = (q @ k.transpose(-2, -1)) * self.scale
        attn_s = attn_qk.softmax(dim=-1)
        attn = self.attn_drop(attn_s)
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_qk
        
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

class Block_qk_map(nn.Module):
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
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_qk_map(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop_ratio)
        
    def forward(self, x):
        x_attn, qk_attn = self.attn(self.norm1(x)) # v4-1版本及以前 有BUG  x应该直接是输入+Attention Block后
        x = x + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, qk_attn

class VisionTransformer_pcc(nn.Module):
    def __init__(self, attn_list, feat_block, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, isEmbed=True):

        super().__init__()
        self.scale = 768 ** -0.5
        self.attn_list = attn_list
        self.feat_block_id = feat_block
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
        self.blocks =  nn.ModuleList([
            Block_qk_map(dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                     norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.norm_middle = norm_layer(embed_dim)
        self.isEmbed = isEmbed
        # self.patch_lin_prj = nn.Linear(embed_dim, embed_dim)
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
        # self.head = nn.Linear(
        #     self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.head_drop = nn.Dropout(p=0.2)
        self.head = nn.Linear(embed_dim*2, num_classes) if num_classes > 0 else nn.Identity()
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
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(
                x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        attn_map_group = []
        for i,blk in enumerate(self.blocks):
            x, attn_map = blk(x)
            if i  in self.attn_list:
                # 将qk_map添加到list
                attn_map_group.append(attn_map)
            # 这里把选中的 block feat返回
            if i == self.feat_block_id and self.feat_block_id != len(self.blocks)-1:
                x_block = x   # [B,197,768]
        x = self.norm(x)
        if self.feat_block_id != len(self.blocks)-1:
            # 如果选取的block不是最后一层则需要norm一下
            x_block = self.norm_middle(x_block)
        else:
            x_block = x
        if len(attn_map_group) > 0:
            attn_block = torch.cat(attn_map_group, dim=1)
        else:
            attn_block = torch.zeros(x.shape[0])
        if self.dist_token is None:
            # 返回选中的Block
            return self.pre_logits(x[:, 0]), x_block[:,1:], x[:, 1:], attn_block
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x, patch_token, last_patch_token, attn_block = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            # # 这里可以提取平均Attention map
            # # patch token [B, 196, 768]
            B, PP, C = patch_token.shape
            attn_qk_map_avg = torch.mean(attn_block, dim=1, keepdim=False) # [B,1,PP+1,PP+1] 在这里断开梯度
            attn_patch_qk_map = attn_qk_map_avg[:, 1:, 1:] # 计算平均patch token之间的qk map [B,196,196]
            localization_map = torch.softmax(attn_patch_qk_map, dim=-1) # [B,196,196] 
            localization_map = convert_consis_3(localization_map) # [B,196,196]
            localization_map = nn.functional.interpolate(localization_map.unsqueeze(1),size=(14,14),mode='bilinear',align_corners=False) # [B,1,14,14]
            localization_map = localization_map.reshape(B,1,PP)  
            x = torch.cat([x, torch.bmm(localization_map, patch_token).squeeze(1).detach()], -1)  # [B,768*2]   # 在这里断开梯度
            # # 直接送入head
            x = self.head(x)
        return x, last_patch_token
    
class VisionTransformer_base(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, isEmbed=True):

        super().__init__()
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
            # 还需要返回其它[B,196,768]的特征token
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

class VisionTransformer_lts(nn.Module):
    def __init__(self, feat_block, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, isEmbed=True):

        super().__init__()
        self.scale = 768 ** -0.5
        assert feat_block != depth
        self.feat_block_id = feat_block
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
        self.blocks =  nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                     norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.norm_middle = norm_layer(embed_dim)
        self.isEmbed = isEmbed
        # self.patch_lin_prj = nn.Linear(embed_dim, embed_dim)
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
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.head_drop = nn.Dropout(p=0.2)
        # self.head = nn.Linear(embed_dim*2, num_classes) if num_classes > 0 else nn.Identity()
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
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(
                x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for i,blk in enumerate(self.blocks):
            x = blk(x)
            # 这里把选中的 block feat返回
            if i == self.feat_block_id:
                x_block = x   # [B,197,768]
        x = self.norm(x)
        if self.feat_block_id != len(self.blocks)-1:
            # 如果选取的block不是最后一层则需要norm一下
            x_block = self.norm_middle(x_block)
        else:
            x_block = x
        if self.dist_token is None:
            # 返回CLS, 中间层Patch, 最后层Patch
            return self.pre_logits(x[:, 0]), x_block[:,1:], x[:, 1:]
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x, mid_patch_token, last_patch_token = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            # # 这里可以提取平均Attention map
            # # patch token [B, 196, 768]
            # B, PP, C = patch_token.shape
            # # 直接送入head
            x = self.head(x)
        return x, mid_patch_token, last_patch_token
    
def convert_consis(map):
    B, _, __ = map.shape
    consis_img = torch.zeros(B, 196, 196).to(map.device)
    for i in range(196):
        start_h = 14*(i // 14)
        start_w = 14*(i % 14)
        for j in range(196):
            h = j // 14
            w = j % 14
            consis_img[:,start_h+h,start_w+w] = map[:,i,j]
    return consis_img

def convert_consis_2(map):
    # map = map.reshape((1,)+map.shape)
    B, PP, _ = map.shape
    map_gpu = map.device
    map = map.cpu().data.numpy()
    consis_img = np.zeros((B,196,196))
    for i in range(196):
        start_h = 14*(i // 14)
        start_w = 14*(i % 14)
        for j in range(196):
            h = j // 14
            w = j % 14
            consis_img[:,start_h+h,start_w+w] = map[:,i,j]
    return torch.tensor(consis_img).to(map_gpu).float()

def convert_consis_3(map):
    # map = map.reshape((1,)+map.shape)
    B, PP, _ = map.shape
    map_gpu = map.device
    map = map.cpu().data.numpy()
    consis_img = np.zeros((B,196,196))
    for i in range(196):
        start_h = 14*(i // 14)
        start_w = 14*(i % 14)
        # for j in range(196):
        #     h = j // 14
        #     w = j % 14
        #     consis_img[:,start_h+h,start_w+w] = map[:,i,j]
        consis_img[:,start_h:start_h+14,start_w:start_w+14] = map[:,i,:].reshape(-1,14,14)
    return torch.tensor(consis_img).to(map_gpu).float()
    
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

def vit_base_patch16_224_in21k_base(num_classes: int = 21843, has_logits: bool = True, isEmbed: bool = True, drop_ratio: float = 0.,
                                           attn_drop_ratio: float = 0., drop_path_ratio: float = 0., keepEmbedWeight: bool = True):
    model = VisionTransformer_base(img_size=224,
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
    # # 删除不需要的权重
    del_keys = ['head.weight', 'head.bias'] if model.has_logits \
        else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    # # # for DEBUG
    # weight_pth = 'output/3090_pretrained/ViT-4/8_0.9926_val.tar'
    # weights_dict = torch.load(weight_pth)["model"]
    print(model.load_state_dict(weights_dict, strict=False))
    return model

def vit_base_patch16_224_in21k_pcc(num_classes: int = 21843, has_logits: bool = True, isEmbed: bool = True, drop_ratio: float = 0.,
                                           attn_drop_ratio: float = 0., drop_path_ratio: float = 0., keepEmbedWeight: bool = True, attn_list: list = [7,8,9,10,11], feat_block: int = 6):
    model = VisionTransformer_pcc(img_size=224,
                                          patch_size=16,
                                          embed_dim=768,
                                          depth=12,
                                          num_heads=12,
                                          drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=drop_path_ratio,
                                          representation_size=768 if has_logits else None,
                                          num_classes=num_classes, isEmbed=isEmbed,
                                          attn_list=attn_list,
                                          feat_block=feat_block)
    if not keepEmbedWeight:
        del model.patch_embed
    weight_pth = 'src/jx_vit_base_patch16_224_in21k-e5005f0a.pth'
    weights_dict = torch.load(weight_pth)
    # # 删除不需要的权重
    del_keys = ['head.weight', 'head.bias'] if model.has_logits \
        else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    # # # for DEBUG
    # weight_pth = 'output/3090_pretrained/ViT-4/8_0.9926_val.tar'
    # weights_dict = torch.load(weight_pth)["model"]
    print(model.load_state_dict(weights_dict, strict=False))
    return model

def vit_base_patch16_224_in21k_lts(num_classes: int = 21843, has_logits: bool = True, isEmbed: bool = True, drop_ratio: float = 0.,
                                           attn_drop_ratio: float = 0., drop_path_ratio: float = 0., keepEmbedWeight: bool = True, feat_block: int = 6):
    model = VisionTransformer_lts(img_size=224,
                                          patch_size=16,
                                          embed_dim=768,
                                          depth=12,
                                          num_heads=12,
                                          drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=drop_path_ratio,
                                          representation_size=768 if has_logits else None,
                                          num_classes=num_classes, isEmbed=isEmbed,
                                          feat_block=feat_block)
    if not keepEmbedWeight:
        del model.patch_embed
    weight_pth = 'src/jx_vit_base_patch16_224_in21k-e5005f0a.pth'
    weights_dict = torch.load(weight_pth)
    # # 删除不需要的权重
    del_keys = ['head.weight', 'head.bias'] if model.has_logits \
        else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    # # # for DEBUG
    # weight_pth = 'output/3090_pretrained/ViT-4/8_0.9926_val.tar'
    # weights_dict = torch.load(weight_pth)["model"]
    print(model.load_state_dict(weights_dict, strict=False))
    return model

'''Xcep Entry here'''
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block_Xcep(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super().__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


def add_gaussian_noise(ins, mean=0, stddev=0.2):
    noise = ins.data.new(ins.size()).normal_(mean, stddev)
    return ins + noise

class Xception_Entry(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, inc=3):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super().__init__()
        # Entry flow
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here
        self.block1 = Block_Xcep(
            64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block_Xcep(
            128, 256, 2, 2, start_with_relu=True, grow_first=True)
        '''此处为了ViT修改到768个通道'''
        self.block3 = Block_Xcep(
            256, 768, 2, 2, start_with_relu=True, grow_first=True)

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------

    def fea_part1_0(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

    def fea_part1_1(self, x):

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def fea_part2(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def forward(self, input):
        x = self.fea_part1_0(input)
        x = self.fea_part1_1(x)
        x = self.fea_part2(x)
        return x

class hXcep(nn.Module):
    def __init__(self):
        self.region_layer = RegionLayerDW(32,32,(7,7))
        self.entry = Xception_Entry()
    def forward(self, input):
        return 
    
def hDRMLPv2_embed(weight_pth:str):
    model = hDRMLPv2Embed(img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d)
    if weight_pth is not None:
        weights_dict = torch.load(weight_pth)['model']
        # # 删除不需要的权重
        keys = list(weights_dict.keys())
        for k in keys:
            if 'hproj' not in k:
                del weights_dict[k]
            else:
                print('Load: ',k)
        print(model.load_state_dict(weights_dict, strict=False))
        
    return model

class PCC_ViT(nn.Module):
    '''第12层多头KQ映射 平均Attn用来Assemble第k层'''
    def __init__(self, weight_pth=None, attn_list=[4,5,6],feat_block=6):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_pcc(
            num_classes=2, has_logits=False, isEmbed=False, keepEmbedWeight=False, attn_list=attn_list, feat_block=feat_block)
        self.custom_embed = hDRMLPv2_embed(weight_pth)
        # consis-1
        self.KQ = nn.Linear(768, 768*2)
        self.scale = 768 ** -0.5
       
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token = self.vit_model(x)
        B, PP, C = patch_token.shape  # [B, 196, 768]
        kq = self.KQ(patch_token).reshape(B, PP, 2, 2, C//2).permute(2, 0, 3, 1, 4)    # reshape参数:[batch,patch token, qk_num, head_nums, c//head_nums]
        k, q = kq[0], kq[1]  # [B,2,PP,C//2]
        # # consis-1
        attn_map = k@q.transpose(-2, -1)* self.scale
        # 这里在损失函数中用sigmoid激活
        consis_map_main = attn_map[:,0]
        consis_map_edge = attn_map[:,1]
        return cls_token, consis_map_main, consis_map_edge

    def test_time(self, x):
        x = self.custom_embed(x)
        cls_token, _ = self.vit_model(x)
        return cls_token
    
class PCC_ViT_dev1(nn.Module):
    '''第12层多头KQ映射 平均Attn用来Assemble第k层'''
    def __init__(self, weight_pth=None, attn_list=[4,5,6],feat_block=6):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_pcc(
            num_classes=2, has_logits=False, isEmbed=False, keepEmbedWeight=False, attn_list=attn_list, feat_block=feat_block)
        # self.custom_embed = hDRMLPv2_embed(weight_pth)
        self.custom_embed = Xception_Entry()
        # consis-1
        self.KQ = nn.Linear(768, 768*2)
        self.scale = 768 ** -0.5
       
    def forward(self, x):
        # x = self.custom_embed(x)
        x = self.custom_embed(x)
        B, C, H, W = x.shape
        x = x.reshape(B,C,-1).permute(0,2,1).contiguous()
        cls_token, patch_token = self.vit_model(x)
        B, PP, C = patch_token.shape  # [B, 196, 768]
        kq = self.KQ(patch_token).reshape(B, PP, 2, 2, C//2).permute(2, 0, 3, 1, 4)    # reshape参数:[batch,patch token, qk_num, head_nums, c//head_nums]
        k, q = kq[0], kq[1]  # [B,2,PP,C//2]
        # # consis-1
        attn_map = k@q.transpose(-2, -1)* self.scale
        # 这里在损失函数中用sigmoid激活
        consis_map_main = attn_map[:,0]
        consis_map_edge = attn_map[:,1]
        return cls_token, consis_map_main, consis_map_edge

    def test_time(self, x):
        x = self.custom_embed(x)
        B, C, H, W = x.shape
        x = x.reshape(B,C,-1).permute(0,2,1).contiguous()
        cls_token, _ = self.vit_model(x)
        return cls_token 
    
class ViT_ImageNet(nn.Module):
    def __init__(self, weight_pth=None):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_base(
            num_classes=100, has_logits=False, isEmbed=False, keepEmbedWeight=False, drop_ratio=0.1)
        # self.custom_embed = hXcep(weight_pth)
        self.custom_embed = Xception_Entry()
    def forward(self, x):
        x = self.custom_embed(x)
        B, C, H, W = x.shape
        x = x.reshape(B,C,-1).permute(0,2,1).contiguous()
        cls_token,_ = self.vit_model(x)
        return cls_token
    def test_time(self, x):
        return self.forward(x)

class PCC_ViT_dev2(nn.Module):
    ''''''
    def __init__(self,weight_pth=None,feat_block=6):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_lts(
            num_classes=1, has_logits=False, isEmbed=False, keepEmbedWeight=False, feat_block=feat_block)
        # self.custom_embed = hDRMLPv2_embed(weight_pth)
        self.custom_embed = hDRMLPv2_embed(weight_pth)
        # consis-1
        self.KQ = nn.Linear(768, 768*2)
        self.scale = 768 ** -0.5
        # LTS
        self.lts_head = nn.Linear(768, 1) # out:[B, 196, 2]
        self.max_pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        '''在中间层做main map监督 最后一层做LTS监督'''
        x = self.custom_embed(x)
        cls, mid_patch_token, last_patch_token = self.vit_model(x)
        # print(cls.shape)
        B, PP, C = mid_patch_token.shape  # [B, 196, 768]
        kq = self.KQ(mid_patch_token).reshape(B, PP, 2, 1, C//1).permute(2, 0, 3, 1, 4)    # reshape参数:[batch, patch token, qk_num, head_nums, c//head_nums]
        k, q = kq[0], kq[1]  # [B,n_head,PP,C//n_head]
        # # consis-1 # 这里在损失函数中用sigmoid激活
        attn_map = k@q.transpose(-2, -1)* self.scale
        consis_map_main = attn_map[:,0]
        # LTS: shared FC
        lts_map = self.lts_head(last_patch_token)  # out:[B, 196, 1]
        lts_feat = 0.5*self.max_pool(lts_map.transpose(-2,-1)).view(B,-1)
        y = cls + lts_feat
        return y.view(B), consis_map_main, lts_map.view(B,-1)

    def test_time(self, x):
        x = self.custom_embed(x)
        cls, _, last_patch_token = self.vit_model(x)
        B, PP, C = last_patch_token.shape  # [B, 196, 768]
        lts_map = self.lts_head(last_patch_token)  # out:[B, 196, 1]
        lts_feat = 0.5*self.max_pool(lts_map.transpose(-2,-1)).view(B,-1)
        y = cls + lts_feat
        return y.view(B)

if __name__ == '__main__':
    from torchinfo import summary
    model = PCC_ViT_dev2()
    # model = ViT_ImageNet()
    # model = Xception_Entry()
    # print(model.vit_model.blocks)
    # print(model)
    image_size = 224
    batch_size = 3
    # input_s = (batch_size, 3, image_size, image_size)
    # summary(model, input_s)
    dummy = torch.rand(batch_size, 3, image_size, image_size)
    y, consis_map, lts_map = model(dummy)
    print(y.shape,consis_map.shape,lts_map.shape)
    # weight_name = 'output/a5000/ViT-hDRMLPv2-UIAv3-1/51_0.9897_val.tar'
    # cnn_sd = torch.load(weight_name)["model"]
    # # print(model.load_state_dict(cnn_sd))
    # model.eval()
    # attn_w = model.vit_model.attn_fusion
    # print(attn_w, torch.softmax(attn_w,dim=-1).data.numpy())
    '''
    for name, para in model.named_parameters():
        # 除head, pre_logits外 其他权重全部冻结
        if "head" not in name and "pre_logits" not in name:
            para.requires_grad_(False)
        else:
            print("training {}".format(name))
    '''
