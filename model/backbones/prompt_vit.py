'''
@File    :   prompt_vit.py
@Time    :   2023/02/18 17:29:40
@Author  :   liyuke 
'''
import math
from functools import partial
from itertools import repeat
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc as container_abcs
from einops import rearrange
from model.backbones.mix_style_algos import DomainMix,EFDMix,DomainQueue,MixHistogram,MixStyle,Mixup

from model.backbones.vit_pytorch import Block, PatchEmbed_overlap, trunc_normal_

int_classes = int
string_classes = str


def resize_pos_embed(posemb, posemb_new, hight, width, prompt_len=None):
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    # posemb_token = posemb_token.expand(-1, prompt_len+1, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


class prompt_vit(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, prompt_length=4, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed_overlap(
            img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
            embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        ##### for domain prompt
        num_domains = kwargs['num_domains']
        self.num_domains = num_domains
        self.prompt_length = prompt_length
        print("=============number of domains = {}=============".format(num_domains))
        self.domain_tokens = nn.Parameter(torch.zeros(num_domains, prompt_length, embed_dim))
        # self.domain_tokens = nn.Parameter(torch.zeros(1, prompt_length, embed_dim))
        # self.prompts_weight = nn.Linear(embed_dim, num_domains, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.mixstyle = MixStyle() #### mixstyle
        self.efdmix = EFDMix()     #### efdmix

        self.domainmix = nn.ModuleList([DomainMix(embed_dim, self.num_domains) for _ in range(depth)])
        print("==using domainmixup==")

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.depth = depth
        self.norm = norm_layer(embed_dim)

        # Classifier head
        # self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.domain_tokens, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, domain=None, stage=None):
        B = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        # if self.training:
        #     domain_tokens = self.domain_tokens[domain].expand(B,-1,-1)
        # else:
        #     #### combine domain tokens (effective)
        #     domain_tokens = self.domain_tokens.mean(dim=0).expand(B,-1,-1)
        # x = torch.cat((x, domain_tokens), dim=1)

        x = self.pos_drop(x)

        for i,blk in enumerate(self.blocks):
            if i < 3:
                #### mix image and domain
                x[:, 1:] = self.mixstyle(x[:, 1:])
                # #### only mix domain (collapse)
                # x[:, -1:] = self.mixstyle(x[:, -1:])
               
                # #### efdmix
                # x[:, -1:] = self.efdmix(x[:, -1:])

                ### domainmix
                x[:, 1:] = self.domainmix[i](x[:, 1:], domain)
                # if random.random() > 0.5:
                # x[:, 1:-self.prompt_length] = nn.InstanceNorm1d(768, affine=False)(x[:, 1:-self.prompt_length])
            x = blk(x)

        x = self.norm(x)

        # return x[:, 0]
        return x # (B, N, C)

    def forward(self, x, domain=None, stage=None):
        x = self.forward_features(x, domain=domain, stage=stage)
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
            # if 'cls' in k: ####### add cls param to domain tokens
            #     self.domain_tokens.data.copy_(v.expand(self.num_domains, self.prompt_length, -1))
            #     count += 1
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x, self.prompt_length)
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



class mix_vit(nn.Module):
    """ Transformer-based Object Re-Identification
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, stem_conv=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed_overlap(
            img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        num_domains = kwargs['num_domains']
        self.num_domains = num_domains

        # self.mixstyle = MixStyle()
        # self.efdmix = EFDMix()
        # self.mixhm = MixHistogram()
        # self.mixup = Mixup()
        self.domainmix = nn.ModuleList([
            DomainMix(embed_dim, num_domains) for _ in range(4)
            ])
        # self.domainmix = DomainMix(embed_dim, num_domains)
        # self.domainqueue = nn.ModuleList([
        #     DomainQueue(embed_dim, num_domains) for _ in range(3)
        #     ])

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.depth = depth
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, labels=None, domain=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        # #### mixup
        # x, y = self.mixup(x, labels)
        # if y is not None: labels = y
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        x = self.pos_drop(x)

        tri_loss_avg = torch.tensor(0.0, device=x.device)
        # count = 0
        # layer_wise_tokens = []
        for i, blk in enumerate(self.blocks):
            if i < 3: #### best 3/12
                # #### mixup
                # x, y = self.mixup(x, labels)
                # if y is not None: labels = y
                
                # #### mixstyle (skip cls token)
                # x[:, 1:] = self.mixstyle(x[:, 1:])

                #### efdmix
                # x = self.efdmix(x)
                # #### efdmix (skip cls token)
                # x[:, 1:] = self.efdmix(x[:, 1:])

                # #### domainmix (skip cls token)
                # x[:, 1:], tri_loss = self.domainmix[i](x[:, 1:], labels, domain)
                # if tri_loss != 0:
                #     count += 1
                #     tri_loss_avg += tri_loss
                #### domainmix
                x[:, 1:], tri_loss = self.domainmix[i](x[:, 1:], labels, domain)

                # #### domainqueue (skip cls token)
                # x[:, 1:] = self.domainqueue[i](x[:, 1:], domain)

                # #### domainqueue
                # x = self.domainqueue[i](x, domain)
            x = blk(x)
            # layer_wise_tokens.append(x)
        # if count != 0:
        #     tri_loss_avg = tri_loss_avg / count

        x = self.norm(x)

        x, tri_loss_avg = self.domainmix[-1](x, labels, domain, True)

        return x, tri_loss_avg
        # return x
    
        # layer_wise_tokens = [self.norm(t) for t in layer_wise_tokens]
        # rand_num = random.randint(0, 11)
        # hint_loss = F.mse_loss(
        #             F.normalize(layer_wise_tokens[rand_num][:, 0],dim=1),
        #             F.normalize(layer_wise_tokens[-1][:, 0],dim=1),
        #             reduction='sum'
        #         ) / B
        # return x[:, 0]
        # return x # (B, N, C)
        # loss = tri_loss_avg + hint_loss
        # return x, loss

    def forward(self, x, labels=None, domain=None):
        x, tri_loss_avg = self.forward_features(x, labels=labels, domain=domain)
        return x, tri_loss_avg
        # x = self.forward_features(x, labels=labels, domain=domain)
        # return x

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






def vit_large_patch16_224_prompt_vit(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm='LN', **kwargs):
    model = prompt_vit(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,\
        drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_name=norm, **kwargs)

    return model

def vit_base_patch16_224_prompt_vit(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm='LN', **kwargs):
    model = prompt_vit(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,\
        drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_name=norm, **kwargs)

    return model

def vit_base_patch16_224_mix_vit(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm='LN', **kwargs):
    model = mix_vit(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,\
        drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_name=norm, **kwargs)

    return model

def vit_base_patch32_224_prompt_vit(img_size=(256, 128), stride_size=32, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm='LN', **kwargs):
    model = prompt_vit(
        img_size=img_size, patch_size=32, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,\
        drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_name=norm, **kwargs)

    return model

def vit_small_patch16_224_prompt_vit(img_size=(256, 128), stride_size=16, drop_rate=0., attn_drop_rate=0.,drop_path_rate=0.1, norm='LN', **kwargs):
    kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = prompt_vit(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=8, num_heads=8,  mlp_ratio=3., qkv_bias=False, drop_path_rate = drop_path_rate,\
        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_name=norm, **kwargs)

    return model

def deit_small_patch16_224_prompt_vit(img_size=(256, 128), stride_size=16, drop_path_rate=0.0, drop_rate=0.0, attn_drop_rate=0.0, norm='LN', **kwargs):
    model = prompt_vit(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, norm_name=norm, **kwargs)

    return model

def deit_tiny_patch16_224_prompt_vit(img_size=(256, 128), stride_size=16, drop_path_rate=0.0, drop_rate=0.0, attn_drop_rate=0.0, norm='LN', **kwargs):
    model = prompt_vit(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, norm_name=norm, **kwargs)

    return model