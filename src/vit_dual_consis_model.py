from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import math

'''这个是基于UIA修改的一致性监督，修复attn map的一些bug'''

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

class VisionTransformer_dual_consisv1(nn.Module):
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
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
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
            if i == self.feat_block_id:
                x_block = x   # [B,197,768]
        x_block = self.norm_middle(x_block)
        x = self.norm(x)
        if len(attn_map_group) > 0:
            attn_block = torch.cat(attn_map_group, dim=1)
        else:
            attn_block = torch.zeros(x.shape[0])
        if self.dist_token is None:
            # 还需要返回其它[B,196,768]的特征token(中间层+最后一层)和attn_block
            return self.pre_logits(x[:, 0]), x_block[:,1:], x[:,1:], attn_block
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x, patch_token_middle, patch_token_last, attn_block = self.forward_features(x)
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
            # attn_qk_map_avg = torch.mean(attn_block, dim=1, keepdim=True)
            # attn_patch_qk_map = attn_qk_map_avg[:, :, 1:, 1:].squeeze(1) # 计算平均patch token之间的qk map [B,196,196]
            # attn_cls_qk_map = attn_qk_map_avg[:, :, 0, 1:]
            # localization_map = torch.softmax(attn_cls_qk_map, dim=-1) # 计算CLS和其它token的平均attn map [B,1,196]
            # localization_map = localization_map.reshape(B,1,PP)
            # x = torch.cat([x, torch.bmm(localization_map, patch_token).squeeze(1)], -1) 
            # # 直接送入head
            x = self.head(x)
        return x, patch_token_middle, patch_token_last

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
        
def vit_base_patch16_224_in21k_dual_consisv1(num_classes: int = 21843, has_logits: bool = True, isEmbed: bool = True, drop_ratio: float = 0.,
                                           attn_drop_ratio: float = 0., drop_path_ratio: float = 0., keepEmbedWeight: bool = True, attn_list: list = [7,8,9,10,11], feat_block: int = 6):
    model = VisionTransformer_dual_consisv1(img_size=224,
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

class Vit_hDRMLPv2_dual_consisv1(nn.Module):
    def __init__(self, weight_pth=None, attn_list=[], feat_block=5):
        super().__init__()
        self.vit_model = vit_base_patch16_224_in21k_dual_consisv1(
            num_classes=2, has_logits=False, isEmbed=False, keepEmbedWeight=False, attn_list=attn_list, feat_block=feat_block)
        self.custom_embed = hDRMLPv2_embed(weight_pth)
        # consis-1
        self.K = nn.Linear(768, 768)
        self.Q = nn.Linear(768, 768)
        self.K2 = nn.Linear(768, 768)
        self.Q2 = nn.Linear(768, 768)
        self.scale = 768 ** -0.5
       
    def forward(self, x):
        x = self.custom_embed(x)
        cls_token, patch_token_middle, patch_token_last = self.vit_model(x)
        # # consis-1
        consis_map_middle = (self.K(patch_token_middle) @
                      self.Q(patch_token_middle).transpose(-2, -1)) * self.scale
        # v4-1里，是共用一个QK的
        # consis_map_last = (self.K(patch_token_last) @
        #               self.Q(patch_token_last).transpose(-2, -1)) * self.scale
        # dual consis里，QK分开
        consis_map_last = (self.K2(patch_token_last) @
                      self.Q2(patch_token_last).transpose(-2, -1)) * self.scale
        return cls_token, consis_map_middle, consis_map_last


    def test_time(self, x):
        x = self.custom_embed(x)
        cls_token, _, _ = self.vit_model(x)
        return cls_token
    
if __name__ == '__main__':
    from torchinfo import summary
    model = Vit_hDRMLPv2_dual_consisv1()
    # print(model.vit_model.blocks)
    # print(model)
    image_size = 224
    batch_size = 1
    input_s = (batch_size, 3, image_size, image_size)
    # # summary(model, input_s)
    dummy = torch.rand(batch_size, 3, image_size, image_size)
    cls_token, consis_map = model(dummy)
    # print(consis_map.shape)
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
