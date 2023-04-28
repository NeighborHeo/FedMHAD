import sys
# gradcam (get the attention maps)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

from collections import OrderedDict 
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import timm
from timm.models.layers import trunc_normal_, DropPath
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from functools import partial
from time import time

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.excluded_heads = None

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attn=False):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # ----------------------------------
        if self.excluded_heads is not None:
            attn_mask = torch.ones_like(attn)
            for head_idx in self.excluded_heads:
                attn_mask[:, head_idx, :, :] = 0
            attn = attn * attn_mask
        # ----------------------------------

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attn:
            return x, attn
        else:
            return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, return_attn=False):
        if return_attn:
            val, attn = self.attn(self.norm1(x), return_attn=True)
            x = x + self.drop_path1(val)
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path1(self.attn(self.norm1(x)))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
            return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, 
                 norm_layer=None, act_layer=None, block_fn=Block):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth
        
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # for gradcam
        self.target_layer = [self.blocks[-1].norm1]
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_attn=False):
        B = x.shape[0]

        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            # if last block
            if blk == self.blocks[-1]:
                x, attn = blk(x, return_attn=True)
            else:
                x = blk(x)
        x = self.norm(x)
        
        x = self.head(x[:, 0])
        
        if return_attn:
            return x, attn
        
        return x

    def setExcludedHead(self, excluded_head):
        for block in self.blocks:
            block.attn.excluded_heads = excluded_head

    def get_class_activation_map(self, x, y):
        targets = None 
        def reshape_transform(tensor, height=14, width=14):
            result = tensor[:, 1:, :].reshape(tensor.size(0),
                                            height, width, tensor.size(2))
            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.transpose(2, 3).transpose(1, 2)
            return result
        cam = GradCAM(model=self, target_layers=self.target_layer, use_cuda=True, reshape_transform=reshape_transform)
        cam.batch_size = 16
        grayscale_cam = cam(input_tensor=x, targets=targets, aug_smooth=True, eigen_smooth=True)
        return grayscale_cam

    @torch.no_grad()
    def get_attention_maps(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn_maps = []
        for blk in self.blocks:
            _, attn_map = blk.attn(x, return_attn=True)
            attn_maps.append(attn_map)
            x = blk(x)
        return attn_maps

    def get_attention_maps_postprocessing(self, x):
        # x.shape = (batch, channel, height, width) = 16, 3, 197, 197
        mha = self.get_attention_maps(x)[-1]
        # print(mha.shape)
        # # mha.shape = (batch, head, height, width) = 16, 3, 197, 197
        # w_featmap = x.shape[-2] // 16
        # h_featmap = x.shape[-1] // 16
        # n_Img, nh, _, _ = mha.shape
        # # w_featmap = 14, h_featmap = 14, nh = 3
        # # we keep only the output patch attention
        # mha = mha[:, :, 0, 1:].reshape(n_Img, nh, -1)
        # # mha.shape = 3, 196(14*14)
        # mha = mha.reshape(n_Img, nh, w_featmap, h_featmap)
        # # mha = nn.functional.interpolate(mha, scale_factor=16, mode="nearest")
        return mha

    def get_attention_maps_postprocessing_(self, x):
        # x.shape = (batch, channel, height, width) = 16, 3, 197, 197
        mha = self.get_attention_maps(x)[-1]
        
        # mha.shape = (batch, head, height, width) = 16, 3, 197, 197
        w_featmap = x.shape[-2] // 16
        h_featmap = x.shape[-1] // 16
        n_Img, nh, _, _ = mha.shape
        # w_featmap = 14, h_featmap = 14, nh = 3
        # we keep only the output patch attention
        mha = mha[:, :, 0, 1:].reshape(n_Img, nh, -1)
        # mha.shape = 3, 196(14*14)
        # we keep only a certain percentage of the mass
        th_attn_list = []
        for img in range(n_Img):
            _mha = mha[img].clone()
            val, idx = torch.sort(_mha)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - 0.6)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=16, mode="nearest")[0]
            th_attn_list.append(th_attn)
        th_attn = torch.stack(th_attn_list, dim=0).cpu().numpy()
        mha = mha.reshape(n_Img, nh, w_featmap, h_featmap)
        mha = nn.functional.interpolate(mha, scale_factor=16, mode="nearest").cpu().numpy()
        return mha, th_attn

def vit_tiny_patch16_224(pretrained=False, patch_size=16, num_heads=3, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_heads=num_heads, embed_dim=192, depth=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        #     map_location="cpu", check_hash=True
        # )
        # state_dict = checkpoint["model"]
        # temp = OrderedDict()
        # for i, j in state_dict.items():   # search all key from model
        #     name = i.replace("head.","")  # change key that doesn't match
        #     temp[name] = j
        # model.load_state_dict(temp, strict=False)
        num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 20
        
        temp_model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)
        m, u = model.load_state_dict(temp_model.state_dict(), strict=False)
    return model

def vit_small_patch16_224(pretrained=False, patch_size=16, num_heads=6, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_heads=num_heads, embed_dim=384, depth=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        #     map_location="cpu", check_hash=True
        # )
        
        # # model.load_state_dict(checkpoint["model"])
        # state_dict = checkpoint["model"]
        # temp = OrderedDict()
        # for i, j in state_dict.items():   # search all key from model
        #     name = i.replace("head.","")  # change key that doesn't match
        #     temp[name] = j
        # model.load_state_dict(temp, strict=False)
        num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 20
        temp_model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
        m, u = model.load_state_dict(temp_model.state_dict(), strict=False)
    return model

def vit_base_patch16_224(pretrained=False, patch_size=16, num_heads=12, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_heads=num_heads, embed_dim=768, depth=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        #     map_location="cpu", check_hash=True
        # )
        # model.load_state_dict(checkpoint["model"])
        num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 20
        temp_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        m, u = model.load_state_dict(temp_model.state_dict(), strict=False)
    return model
