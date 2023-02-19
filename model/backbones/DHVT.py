'''
Copy from https://github.com/ArieSeirack/DHVT
Bridging the Gap Between Vision Transformers and
Convolutional Neural Networks on Small Datasets
'''
from collections import OrderedDict
from functools import partial
import logging
import math
import torch
import torch.nn as nn
from itertools import repeat
import collections.abc
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.helpers import named_apply, build_model_with_cfg
import torch.functional as F

from model.backbones.vit_pytorch import Block, DropPath, PatchEmbed_overlap, resize_pos_embed, trunc_normal_, to_2tuple

_logger = logging.getLogger("reid")

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    # deit models (FB weights)
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
}


# The proposed DAFF
class DAFF(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 kernel_size=3, with_bn=True, num_patches=(16,8)):
        super().__init__()
        self.num_patches = num_patches
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # pointwise
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0)
        # depthwise
        self.conv2 = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, groups=hidden_features)
        
        # pointwise
        self.conv3 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.act = act_layer()
        
        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.bn2 = nn.BatchNorm2d(hidden_features)
        self.bn3 = nn.BatchNorm2d(out_features)
        
        # The reduction ratio is always set to 4
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Linear(in_features, in_features//4)
        self.excitation = nn.Linear(in_features//4, in_features)
                
    def forward(self, x):
        B, N, C = x.size()
        cls_token, tokens = torch.split(x, [1, N - 1], dim=1)
        # x = tokens.reshape(B, int(math.sqrt(N - 1)), int(math.sqrt(N - 1)), C).permute(0, 3, 1, 2)
        x = tokens.reshape(B, self.num_patches[0], self.num_patches[1], C).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        shortcut = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = shortcut + x

        x = self.conv3(x)
        x = self.bn3(x)

        weight = self.squeeze(x).flatten(1).reshape(B, 1, C)
        weight = self.excitation(self.act(self.compress(weight)))
        cls_token = cls_token * weight
        
        tokens = x.flatten(2).permute(0, 2, 1)
        out = torch.cat((cls_token, tokens), dim=1)
        
        return out


# The proposed HI-MHSA in DHVT
class HI_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.act = nn.GELU()
        self.ht_proj = nn.Linear(dim//self.num_heads, dim,bias=True)
        self.ht_norm = nn.LayerNorm(dim//self.num_heads)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_heads, dim))
        trunc_normal_(self.pos_embed, std=.02)
    
    def forward(self, x):
        B, N, C = x.shape
        H = W =int(math.sqrt(N-1))

        # head token
        head_pos = self.pos_embed.expand(x.shape[0], -1, -1)
        x_ = x.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        x_ = x_.mean(dim=2)  # now the shape is [B, h, 1, d//h]
        x_ = self.ht_proj(x_).reshape(B, -1, self.num_heads, C // self.num_heads)
        x_ = self.act(self.ht_norm(x_)).flatten(2)
        x_ = x_ + head_pos
        x = torch.cat([x, x_], dim=1)
        
        # normal mhsa
        qkv = self.qkv(x).reshape(B, N+self.num_heads, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N+self.num_heads, C)
        x = self.proj(x)
        
        # merge head tokens into cls token
        cls, patch, ht = torch.split(x, [1, N-1, self.num_heads], dim=1)
        cls = cls + torch.mean(ht, dim=1, keepdim=True)
        x = torch.cat([cls, patch], dim=1)

        x = self.proj_drop(x)

        return x


# DHVT Encoder block with HI-MHSA and SOPE
class DHVT_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., qk_scale=None, 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = HI_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DAFF(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, kernel_size=3)
        self.mlp_hidden_dim = mlp_hidden_dim

    def forward(self, x):
        B, N, C = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        ),
        # nn.SyncBatchNorm(out_planes)
        nn.BatchNorm2d(out_planes)
    )

# Affine on Conv-style features with shape of [B, C, H, W]
class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, dim, 1, 1]),requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1, 1]),requires_grad=True)

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x

# The proposed SOPE in DHVT
class ConvPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, init_values=1e-2):
        super().__init__()
        ori_img_size = img_size
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        if patch_size[0] == 16:
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim // 8, 2),
                nn.GELU(),
                conv3x3(embed_dim // 8, embed_dim // 4, 2),
                nn.GELU(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size[0] == 4:  
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size[0] == 2:  
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim, 2),
                nn.GELU(),
            )
        else:
            raise("For convolutional projection, patch size has to be in [2, 4, 16]")
        self.pre_affine = Affine(3)
        self.post_affine = Affine(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape 
        
        x = self.pre_affine(x)
        x = self.proj(x)
        x = self.post_affine(x)

        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)

        return x


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed_overlap, norm_layer=None,
                 act_layer=None, weight_init='',apply_dhvt=True):

        super().__init__()
        self.img_size = img_size
        self.depth = depth
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.apply_dhvt = apply_dhvt
        
        # Patch Embedding
        if self.apply_dhvt:
            self.patch_embed = ConvPatchEmbed(img_size=img_size, embed_dim=embed_dim, patch_size=patch_size)
        else:
            self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1])) if distilled else None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
 
        if self.apply_dhvt:
            self.blocks = nn.ModuleList([
                DHVT_Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
                for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if not self.apply_dhvt:
            trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # @torch.jit.ignore()
    # def load_pretrained(self, checkpoint_path, prefix=''):
    #     _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    # @property
    # def name(self):
    #     return "vit_simple_topk"
    
    
    def forward_features(self, x):
        B, _, h, w = x.shape
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # for input with another resolution, interpolate the positional embedding.
        # used for finetining a ViT on images with larger size.
        # only applied in vanilla ViT
        if not self.apply_dhvt:
            pos_embed = self.pos_embed
            if x.shape[1] != pos_embed.shape[1]:
                assert h == w  # for simplicity assume h == w
                real_pos = pos_embed[:, self.num_tokens:]
                hw = int(math.sqrt(real_pos.shape[1]))
                true_hw = int(math.sqrt(x.shape[1] - self.num_tokens))
                real_pos = real_pos.transpose(1, 2).reshape(1, self.embed_dim, hw, hw)
                new_pos = F.interpolate(real_pos, size=true_hw, mode='bicubic', align_corners=False)
                new_pos = new_pos.reshape(1, self.embed_dim, -1).transpose(1, 2)
                pos_embed = torch.cat([pos_embed[:, :self.num_tokens], new_pos], dim=1)

            x = self.pos_drop(x + pos_embed)
        
        for i, blk in enumerate(self.blocks):
            x  = blk(x)
            
        x = self.norm(x)
        
        if self.dist_token is None:
            return self.pre_logits(x)
        else:
            return x[:, 0], x[:, 1]
        
        
    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        count = 0
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k or 'pre_logits' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
                count += 1
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
        print('Load %d / %d layers.'%(count,len(self.state_dict().keys())))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        # print("Number of parameter: %.2fM" % (total/1e6))
        return total/1e6


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    default_cfg.update(kwargs)
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


# -------------------------------------------------------------
# DHVT model variants

@register_model
def dhvt_tiny_imagenet_patch16(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm='LN', **kwargs):
    model_kwargs = dict(img_size=img_size, patch_size=stride_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, apply_dhvt=True, num_classes=0)
    model_kwargs.update(kwargs)
    model = _create_vision_transformer('deit_tiny_patch16_224', pretrained=False, **model_kwargs)
    _logger.info(model)
    return model


@register_model
def dhvt_small_imagenet_patch16(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm='LN', **kwargs):
    model_kwargs = dict(img_size=img_size, patch_size=stride_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, apply_dhvt=True, num_classes=0)
    model_kwargs.update(kwargs)
    model = _create_vision_transformer('deit_small_patch16_224', pretrained=False, **model_kwargs)
    _logger.info(model)
    return model
