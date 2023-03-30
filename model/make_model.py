import logging
import os
import random
from einops import rearrange
from model.backbones.DHVT import dhvt_small_imagenet_patch16, dhvt_tiny_imagenet_patch16
from model.backbones.mae import PretrainVisionTransformerDecoder, color_vit_decoder, get_sinusoid_encoding_table, mask_vit_decoder, pretrain_mae_base_patch16_224
from model.backbones.normalizations import BatchNorm, InstanceNorm
from model.backbones.prompt_vit import deit_small_patch16_224_prompt_vit, deit_tiny_patch16_224_prompt_vit, vit_base_patch16_224_mix_vit, vit_base_patch16_224_prompt_vit, vit_base_patch32_224_prompt_vit, vit_large_patch16_224_prompt_vit, vit_small_patch16_224_prompt_vit
from model.backbones.swin_transformer import swin_base_patch4_window7_224, swin_small_patch4_window7_224
# from threading import local
from model.backbones.vit_pytorch import DistillViT, TransReID, deit_tiny_patch16_224_TransReID, local_attention_deit_small, local_attention_deit_tiny, local_attention_vit_base, local_attention_vit_base_p32, local_attention_vit_large, local_attention_vit_small, mask_vit_base, vit_base_patch32_224_TransReID, vit_large_patch16_224_TransReID
import torch
import torch.nn as nn

from .backbones.resnet import BasicBlock, ResNet, Bottleneck
from .backbones.resnet_ibn import resnet50_ibn_b,resnet50_ibn_a,resnet101_ibn_b,resnet101_ibn_a
from .backbones import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID

__factory_T_type = {
    'vit_large_patch16_224_TransReID': vit_large_patch16_224_TransReID,
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_base_patch32_224_TransReID': vit_base_patch32_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID,
    'deit_tiny_patch16_224_TransReID': deit_tiny_patch16_224_TransReID,
    'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
    'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
    'dhvt_tiny_patch16': dhvt_tiny_imagenet_patch16,
    'dhvt_small_patch16': dhvt_small_imagenet_patch16
}

__factory_PT_type = {
    'vit_large_patch16_224_prompt_vit': vit_large_patch16_224_prompt_vit,
    'vit_base_patch16_224_prompt_vit': vit_base_patch16_224_prompt_vit,
    'vit_base_patch32_224_prompt_vit': vit_base_patch32_224_prompt_vit,
    'deit_base_patch16_224_prompt_vit': vit_base_patch16_224_prompt_vit,
    'vit_small_patch16_224_prompt_vit': vit_small_patch16_224_prompt_vit,
    'deit_small_patch16_224_prompt_vit': deit_small_patch16_224_prompt_vit,
    'deit_tiny_patch16_224_prompt_vit': deit_tiny_patch16_224_prompt_vit,
}

__factory_LAT_type = {
    'vit_large_patch16_224_TransReID': local_attention_vit_large,
    'vit_base_patch16_224_TransReID': local_attention_vit_base,
    'vit_base_patch32_224_TransReID': local_attention_vit_base_p32,
    'deit_base_patch16_224_TransReID': local_attention_vit_base,
    'vit_small_patch16_224_TransReID': local_attention_vit_small,
    'deit_small_patch16_224_TransReID': local_attention_deit_small,
    'deit_tiny_patch16_224_TransReID': local_attention_deit_tiny,
}

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, model_name, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        
        # model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 2048
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
            model_path = os.path.join(model_path_base, \
                "resnet18-f37072fd.pth")
            print('using resnet18 as a backbone')
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            model_path = os.path.join(model_path_base, \
                "resnet34-b627a593.pth")
            print('using resnet34 as a backbone')
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            model_path = os.path.join(model_path_base, \
                "resnet50-0676ba61.pth")
            print('using resnet50 as a backbone')
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
            model_path = os.path.join(model_path_base, \
                "resnet101-63fe2227.pth")
            print('using resnet101 as a backbone')
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            model_path = os.path.join(model_path_base, \
                "resnet152-394f9c45.pth")
            print('using resnet152 as a backbone')
        elif model_name == 'ibnnet50b':
            self.base = resnet50_ibn_b(pretrained=True)
        elif model_name == 'ibnnet50a':
            self.base = resnet50_ibn_a(pretrained=True)
        elif model_name == 'ibnnet101b':
            self.base = resnet101_ibn_b(pretrained=True)
        elif model_name == 'ibnnet101a':
            self.base = resnet101_ibn_a(pretrained=True)
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet' and 'ibn' not in model_name:
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        # self.pool = nn.Linear(in_features=16*8, out_features=1, bias=False)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, domains=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x) # B, C, h, w
        
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        # global_feat = self.pool(x.flatten(2)).squeeze() # is GAP harming generalization?

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat, label, None
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('reid.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))

# alter this to your pre-trained file name
lup_path_name = {
    'vit_base_patch16_224_TransReID': 'vit_base_ics_cfs_lup.pth',
    'vit_small_patch16_224_TransReID': 'vit_base_ics_cfs_lup.pth',
}

# alter this to your pre-trained file name
imagenet_path_name = {
    'vit_large_patch16_224_TransReID': 'jx_vit_large_p16_224-4ee7a4dc.pth',
    'vit_base_patch16_224_TransReID': 'jx_vit_base_p16_224-80ecf9dd.pth',
    'vit_base_patch32_224_TransReID': 'jx_vit_base_patch32_224_in21k-8db57226.pth', 
    'deit_base_patch16_224_TransReID': 'deit_base_distilled_patch16_224-df68dfff.pth',
    'vit_small_patch16_224_TransReID': 'vit_small_p16_224-15ec54c9.pth',
    'deit_small_patch16_224_TransReID': 'deit_small_distilled_patch16_224-649709d9.pth',
    'deit_tiny_patch16_224_TransReID': 'deit_tiny_distilled_patch16_224-b40b3cf7.pth', 
    'vit_large_patch16_224_prompt_vit': 'jx_vit_large_p16_224-4ee7a4dc.pth',
    'vit_base_patch16_224_prompt_vit': 'jx_vit_base_p16_224-80ecf9dd.pth',
    'vit_base_patch32_224_prompt_vit': 'jx_vit_base_patch32_224_in21k-8db57226.pth',
    'deit_base_patch16_224_prompt_vit': 'deit_base_distilled_patch16_224-df68dfff.pth',
    'vit_small_patch16_224_prompt_vit': 'vit_small_p16_224-15ec54c9.pth',
    'deit_small_patch16_224_prompt_vit': 'deit_small_distilled_patch16_224-649709d9.pth',
    'deit_tiny_patch16_224_prompt_vit': 'deit_tiny_distilled_patch16_224-b40b3cf7.pth',
    'swin_base_patch4_window7_224': 'swin_base_patch4_window7_224_22k.pth', 
    'swin_small_patch4_window7_224': 'swin_small_patch4_window7_224_22k.pth',
}

norm_layer = {
    'LN': nn.LayerNorm,
    'BN': BatchNorm,
    'IN': InstanceNorm,
}

in_plane_dict = {
    'dhvt_tiny_patch16': 192,
    'deit_tiny_patch16_224_TransReID': 192,
    'dhvt_small_patch16': 384,
    'deit_small_patch16_224_TransReID': 384,
    'vit_small_patch16_224_TransReID': 768,
    'deit_base_patch16_224_TransReID': 768,
    'vit_base_patch16_224_TransReID': 768,
    'swin_small_patch4_window7_224': 768,
    'vit_large_patch16_224_TransReID': 1024,
    'swin_base_patch4_window7_224': 1024,
    'vit_large_patch16_224_prompt_vit': 1024,
    'vit_base_patch16_224_prompt_vit': 768,
    'vit_base_patch32_224_prompt_vit': 768,
    'deit_base_patch16_224_prompt_vit': 768,
    'vit_small_patch16_224_prompt_vit': 768,
    'deit_small_patch16_224_prompt_vit': 384,
    'deit_tiny_patch16_224_prompt_vit': 192,
}

class build_vit(nn.Module):
    def __init__(self, num_classes, cfg, factory, num_cls_dom_wise=None):
        super().__init__()
        self.cfg = cfg
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        
        self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if cfg.MODEL.TRANSFORMER_TYPE in in_plane_dict:
            self.in_planes = in_plane_dict[cfg.MODEL.TRANSFORMER_TYPE]
        else:
            print("===== unknown transformer type =====")
            self.in_planes = 768

        print('using Transformer_type: vit as a backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        norm = norm_layer[cfg.MODEL.NORM.TYPE]
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
            (img_size=cfg.INPUT.SIZE_TRAIN,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate= cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            norm=norm)

        if self.pretrain_choice == 'imagenet':
            path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
            self.model_path = os.path.join(model_path_base, path)
            self.base.load_param(self.model_path)
            print('Loading pretrained ImageNet model......from {}'.format(self.model_path))
            
        #### original one
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        # #### multi-domain head
        # if num_cls_dom_wise is not None:
        #     self.classifiers = nn.ModuleList(
        #         nn.Linear(self.in_planes, num_cls_dom_wise[i])\
        #             for i in range(len(num_cls_dom_wise))
        #     )

    def forward(self, x, target=None, domain=None):
        x = self.base(x) # B, N, C
        global_feat = x[:, 0] # cls token for global feature

        feat = self.bottleneck(global_feat)

        if self.training:
            #### original
            cls_score = self.classifier(feat)
            # #### test for ACL
            # global_feat = nn.functional.normalize(feat,2,1)
            # global_feat = feat
            return cls_score, global_feat, target, None

            # #### multi-domain head
            # cls_score = self.classifier(feat)
            # cls_score_ = []
            # for i in range(len(self.classifiers)):
            #     if i not in domain:
            #         cls_score_.append(None)
            #         continue
            #     idx = torch.nonzero(domain==i).squeeze()
            #     cls_score_.append(self.classifiers[i](feat[idx]))
            # return cls_score, global_feat, target, cls_score_
        
            # #### memoryhead (from M3L)
            # return feat, global_feat, target, None

        else:
            return feat if self.neck_feat == 'after' else global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        count = 0
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            # if 'bottleneck' in i:
            #     continue
            if i in self.state_dict().keys():
                self.state_dict()[i].copy_(param_dict[i])
                count += 1
        print('Loading trained model from {}\n Load {}/{} layers'.format(trained_path, count, len(self.state_dict())))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('reid.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))

class build_memory_cls_vit(nn.Module):
    def __init__(self, num_classes, cfg, factory, num_cls_dom_wise=None):
        super().__init__()
        self.cfg = cfg
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        
        self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if cfg.MODEL.TRANSFORMER_TYPE in in_plane_dict:
            self.in_planes = in_plane_dict[cfg.MODEL.TRANSFORMER_TYPE]
        else:
            print("===== unknown transformer type =====")
            self.in_planes = 768

        print('using Transformer_type: vit as a backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        norm = norm_layer[cfg.MODEL.NORM.TYPE]
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
            (img_size=cfg.INPUT.SIZE_TRAIN,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate= cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            norm=norm)
        path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]

        # self.base = ResNet(last_stride=1,
        #                        block=Bottleneck,
        #                        layers=[3, 4, 6, 3])
        # path = "resnet50-0676ba61.pth"
        # self.in_planes = 2048

        if self.pretrain_choice == 'imagenet':
            self.model_path = os.path.join(model_path_base, path)
            self.base.load_param(self.model_path)
            print('Loading pretrained ImageNet model......from {}'.format(self.model_path))
            
        # #### original one
        # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        # self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        # #### multi-domain head
        # if num_cls_dom_wise is not None:
        #     self.classifiers = nn.ModuleList(
        #         nn.Linear(self.in_planes, num_cls_dom_wise[i])\
        #             for i in range(len(num_cls_dom_wise))
        #     )

    def forward(self, x, target=None, domain=None):
        x = self.base(x) # B, N, C
        global_feat = x[:, 0] # cls token for global feature
        # global_feat = x.mean(dim=[2,3]) # for resnet
        feat = self.bottleneck(global_feat)

        if self.training:
            # #### original
            # cls_score = self.classifier(feat)
            # return cls_score, global_feat, target, None

            # #### multi-domain head
            # cls_score = self.classifier(feat)
            # cls_score_ = []
            # for i in range(len(self.classifiers)):
            #     if i not in domain:
            #         cls_score_.append(None)
            #         continue
            #     idx = torch.nonzero(domain==i).squeeze()
            #     cls_score_.append(self.classifiers[i](feat[idx]))
            # return cls_score, global_feat, target, cls_score_
        
            #### memoryhead (from M3L)
            return feat, global_feat, target, None

        else:
            return feat if self.neck_feat == 'after' else global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        count = 0
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            # if 'bottleneck' in i:
            #     continue
            if i in self.state_dict().keys():
                self.state_dict()[i].copy_(param_dict[i])
                count += 1
        print('Loading trained model from {}\n Load {}/{} layers'.format(trained_path, count, len(self.state_dict())))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('reid.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))

class build_prompt_vit(nn.Module):
    def __init__(self, num_classes, cfg, factory):
        super().__init__()
        self.in_planes = in_plane_dict[cfg.MODEL.TRANSFORMER_TYPE]
        self.num_classes = num_classes
        self.neck_feat = cfg.TEST.FEAT_NORM
        self.num_block = cfg.MODEL.DISTILL.NUM_SELECT_BLOCK
        self.if_head = cfg.MODEL.DISTILL.IF_HEAD
        self.num_domains = cfg.DATASETS.NUM_DOMAINS
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
            img_size=cfg.INPUT.SIZE_TRAIN,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate= cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            num_domains=self.num_domains)

        model_path_base = cfg.MODEL.PRETRAIN_PATH
        path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        model_path = os.path.join(model_path_base, path)
        self.base.load_param(model_path)
        print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.bottleneck = nn.BatchNorm1d(self.in_planes) 
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        ##### adpative domain weights
        # self.domain_weight = nn.Linear(self.in_planes*self.base.prompt_length, self.num_domains, bias=False)
        # self.domain_weight = nn.Linear(self.in_planes, self.num_domains, bias=False)
        # self.domain_weight.apply(weights_init_classifier)
        

        self.domain_head = nn.Linear(self.in_planes*self.base.prompt_length, self.num_domains, bias=False)

    def forward(self, x, domain=None):
        B = x.shape[0]
        if self.training:
            x = self.base(x, domain=domain) # B, N, C
            cls_token = x[:, 0] # cls token for global feature

            dom_token = x[:, -self.base.prompt_length:].view(B, -1)
            dom_pre = self.domain_head(dom_token)

            feat = self.bottleneck(cls_token)

            cls_score = self.classifier(feat)
            return cls_score, cls_token, dom_pre
        else:
            x = self.base(x, stage='eval_1') # B, N, C
            cls_token = x[:, 0]
            return cls_token

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        count = 0
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            # if 'bottleneck' in i:
            #     continue
            if i in self.state_dict().keys():
                self.state_dict()[i].copy_(param_dict[i])
                count += 1
        print('Loading trained model from {}\n Load {}/{} layers'.format(trained_path, count, len(self.state_dict())))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('reid.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))

class build_mix_vit(nn.Module):
    def __init__(self, num_classes, cfg, factory):
        super().__init__()
        self.in_planes = in_plane_dict[cfg.MODEL.TRANSFORMER_TYPE]
        self.num_classes = num_classes
        self.neck_feat = cfg.TEST.FEAT_NORM
        self.num_block = cfg.MODEL.DISTILL.NUM_SELECT_BLOCK
        self.if_head = cfg.MODEL.DISTILL.IF_HEAD
        self.num_domains = cfg.DATASETS.NUM_DOMAINS
        self.base = vit_base_patch16_224_mix_vit(
            img_size=cfg.INPUT.SIZE_TRAIN,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate= cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            num_domains=self.num_domains)

        model_path_base = cfg.MODEL.PRETRAIN_PATH
        path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        model_path = os.path.join(model_path_base, path)
        self.base.load_param(model_path)
        print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.bottleneck = nn.BatchNorm1d(self.in_planes) 
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        
    def forward(self, x, labels=None, domains=None):
        x, tri_loss = self.base(x, labels, domains) # B, N, C
        global_feat = x[:, 0] # cls token for global feature

        feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat, labels, None, tri_loss
        else:
            return feat if self.neck_feat == 'after' else global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        count = 0
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            # if 'bottleneck' in i:
            #     continue
            if i in self.state_dict().keys():
                self.state_dict()[i].copy_(param_dict[i])
                count += 1
        print('Loading trained model from {}\n Load {}/{} layers'.format(trained_path, count, len(self.state_dict())))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('reid.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))

class build_distill_vit(nn.Module):
    def __init__(self, num_classes, cfg, factory):
        super().__init__()
        self.in_planes = in_plane_dict[cfg.MODEL.TRANSFORMER_TYPE]
        self.num_classes = num_classes
        self.neck_feat = cfg.TEST.FEAT_NORM
        self.num_block = cfg.MODEL.DISTILL.NUM_SELECT_BLOCK
        self.if_head = cfg.MODEL.DISTILL.IF_HEAD
        self.base = DistillViT(
            img_size=cfg.INPUT.SIZE_TRAIN,
            patch_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate= cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE
            )

        model_path_base = cfg.MODEL.PRETRAIN_PATH
        path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        model_path = os.path.join(model_path_base, path)
        self.base.load_param(model_path)
        print('Loading pretrained ImageNet model......from {}'.format(model_path))
        # 12 bnneck, 12 cls
        self.bottleneck = nn.ModuleList()
        self.classifier = nn.ModuleList()
        for i in range(len(self.base.blocks)):
            bottleneck = nn.BatchNorm1d(self.in_planes) 
            bottleneck.bias.requires_grad_(False)
            bottleneck.apply(weights_init_kaiming)
            self.bottleneck.append(bottleneck)
            classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            classifier.apply(weights_init_classifier)
            self.classifier.append(classifier)
        
    def forward(self, x):
        num_block = self.num_block

        list_out= self.base(x) # blocks, bs, C
        global_feat = list_out[-1]
        feat = self.bottleneck[-1](global_feat)
        if self.training:
            cls_scores = []
            for i in range(self.base.depth):
                feat = self.bottleneck[i](list_out[i])
                cls_scores.append(self.classifier[i](feat))
            output = {'feat': global_feat, 'score': cls_scores[-1]}
            random_block_idx = random.sample(range(self.base.depth-1), num_block)
            output_rbs = []
            for n in random_block_idx:
                output_rbs.append({'feat': list_out[n], 'score': cls_scores[n]})
            return cls_scores, list_out, output, output_rbs, random_block_idx
        else:
            return feat if self.neck_feat == 'after' else global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        count = 0
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            # if 'bottleneck' in i:
            #     continue
            if i in self.state_dict().keys():
                self.state_dict()[i].copy_(param_dict[i])
                count += 1
        print('Loading trained model from {}\n Load {}/{} layers'.format(trained_path, count, len(self.state_dict())))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('reid.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))

# class build_mask_vit(nn.Module):
#     def __init__(self, num_classes, cfg, factory):
#         super().__init__()
#         self.cfg = cfg
#         model_path_base = cfg.MODEL.PRETRAIN_PATH
#         path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
#         self.model_path = os.path.join(model_path_base, path)
#         self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         self.in_planes = 768

#         print('using Transformer_type: vit as a backbone')

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.num_classes = num_classes
#         norm = norm_layer[cfg.MODEL.NORM.TYPE]
#         self.base = mask_vit_base\
#             (img_size=cfg.INPUT.SIZE_TRAIN,
#             stride_size=cfg.MODEL.STRIDE_SIZE,
#             drop_path_rate=cfg.MODEL.DROP_PATH,
#             drop_rate= cfg.MODEL.DROP_OUT,
#             attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
#             norm=norm)
#         if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
#             self.in_planes = 384
#         elif cfg.MODEL.TRANSFORMER_TYPE == 'deit_tiny_patch16_224_TransReID':
#             self.in_planes = 192
#         elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch16_224_TransReID':
#             self.in_planes = 1024
#         if self.pretrain_choice == 'imagenet':
#             self.base.load_param(self.model_path)
#             print('Loading pretrained ImageNet model......from {}'.format(self.model_path))
            
#         self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
#         self.classifier.apply(weights_init_classifier)
#         self.bottleneck = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck.bias.requires_grad_(False)
#         self.bottleneck.apply(weights_init_kaiming)

#     def forward(self, x, mask=None):
#         x = self.base(x, mask) # B, N, C
#         global_feat = x[:, 0] # cls token for global feature

#         feat = self.bottleneck(global_feat)

#         if self.training:
#             cls_score = self.classifier(feat)
#             return cls_score, global_feat
#         else:
#             return feat if self.neck_feat == 'after' else global_feat

#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             if 'classifier' in i: # drop classifier
#                 continue
#             if 'bottleneck' in i:
#                 continue
#             self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#         print('Loading trained model from {}'.format(trained_path))

#     def load_param_finetune(self, model_path):
#         param_dict = torch.load(model_path)
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])
#         print('Loading pretrained model for finetuning from {}'.format(model_path))

#     def compute_num_params(self):
#         total = sum([param.nelement() for param in self.parameters()])
#         logger = logging.getLogger('reid.train')
#         logger.info("Number of parameter: %.2fM" % (total/1e6))

#######
# class build_mae(nn.Module):
#     def __init__(self, num_classes, cfg, factory):
#         super().__init__()
#         self.cfg = cfg
#         model_path_base = cfg.MODEL.PRETRAIN_PATH
#         path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
#         self.model_path = os.path.join(model_path_base, path)
#         self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         self.in_planes = 768

#         print('using Transformer_type: vit as a backbone')

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.num_classes = num_classes
#         norm = norm_layer[cfg.MODEL.NORM.TYPE]
#         self.base = pretrain_mae_base_patch16_224\
#             (pretrained=False,
#             img_size=cfg.INPUT.SIZE_TRAIN,
#             stride_size=cfg.MODEL.STRIDE_SIZE,
#             init_ckpt="/home/nihao/data/checkpoints/mae_pretrain_vit_base.pth")
#         if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
#             self.in_planes = 384
#         elif cfg.MODEL.TRANSFORMER_TYPE == 'deit_tiny_patch16_224_TransReID':
#             self.in_planes = 192
#         elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch16_224_TransReID':
#             self.in_planes = 1024
#         # if self.pretrain_choice == 'imagenet':
#         #     self.base.encoder.load_param(self.model_path)
#         #     print('Loading pretrained ImageNet model......from {}'.format(self.model_path))
            
#         self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
#         self.classifier.apply(weights_init_classifier)
#         self.bottleneck = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck.bias.requires_grad_(False)
#         self.bottleneck.apply(weights_init_kaiming)

#     def forward(self, x, mask=None, vis=False):
#         if not self.training and not vis:
#             x = self.base(x) # B, N, C
#             global_feat = x[:, 0] # cls token for global feature
#             feat = self.bottleneck(global_feat)
#             return feat if self.neck_feat == 'after' else global_feat

#         x, x_ = self.base(x, mask, vis)
#         # x = self.base(x, mask)
#         global_feat = x[:, 0] # cls token for global feature
#         feat = self.bottleneck(global_feat)
#         cls_score = self.classifier(feat)

#         return cls_score, global_feat, x_
#         # return cls_score, global_feat

#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             if 'classifier' in i: # drop classifier
#                 continue
#             if 'bottleneck' in i:
#                 continue
#             self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#         print('Loading trained model from {}'.format(trained_path))

#     def load_param_finetune(self, model_path):
#         param_dict = torch.load(model_path)
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])
#         print('Loading pretrained model for finetuning from {}'.format(model_path))

#     def compute_num_params(self):
#         total = sum([param.nelement() for param in self.parameters()])
#         logger = logging.getLogger('reid.train')
#         logger.info("Number of parameter: %.2fM" % (total/1e6))

# ###### solve DG problem with ssl
# class build_DG_ssl_vit(nn.Module):
#     def __init__(self, num_classes, cfg, factory):
#         super().__init__()
#         self.cfg = cfg
#         model_path_base = cfg.MODEL.PRETRAIN_PATH
#         path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
#         self.model_path = os.path.join(model_path_base, path)
#         self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         self.in_planes = 768

#         print('using Transformer_type: vit as a backbone')

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.num_classes = num_classes
#         norm = norm_layer[cfg.MODEL.NORM.TYPE]
#         ## backbone
#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
#             (img_size=cfg.INPUT.SIZE_TRAIN,
#             stride_size=cfg.MODEL.STRIDE_SIZE,
#             drop_path_rate=cfg.MODEL.DROP_PATH,
#             drop_rate= cfg.MODEL.DROP_OUT,
#             attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
#             norm=norm)
#         if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
#             self.in_planes = 384
#         elif cfg.MODEL.TRANSFORMER_TYPE == 'deit_tiny_patch16_224_TransReID':
#             self.in_planes = 192
#         elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch16_224_TransReID':
#             self.in_planes = 1024
#         if self.pretrain_choice == 'imagenet':
#             self.base.load_param(self.model_path)
#             print('Loading pretrained ImageNet model......from {}'.format(self.model_path))
        
#         ## head & BNNeck
#         self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
#         self.classifier.apply(weights_init_classifier)
#         self.bottleneck = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck.bias.requires_grad_(False)
#         self.bottleneck.apply(weights_init_kaiming)

#         # decoder for img reconstruction
#         if self.in_planes == 3*cfg.MODEL.STRIDE_SIZE**2:
#             self.encoder_to_decoder = nn.Identity()
#         else:
#             self.encoder_to_decoder = nn.Linear(self.in_planes, 3*cfg.MODEL.STRIDE_SIZE**2, bias=False)
#         de_norm = norm_layer[cfg.MODEL.DECODER.NORM]
#         self.decoder = mask_vit_decoder(
#             num_classes=3*cfg.MODEL.STRIDE_SIZE**2,
#             embed_dim=cfg.MODEL.DECODER.DIM,
#             depth=cfg.MODEL.DECODER.DEPTH,
#             num_heads=cfg.MODEL.DECODER.NUM_HEAD,
#             mlp_ratio=cfg.MODEL.DECODER.MLP_RATIO,
#             norm_layer=de_norm
#             # num_classes=3*cfg.MODEL.STRIDE_SIZE**2,
#             # embed_dim=3*cfg.MODEL.STRIDE_SIZE**2,
#             # depth=len(self.base.blocks),
#             # num_heads=self.base.num_heads,
#             # mlp_ratio=self.base.mlp_ratio
#         )
#         self.decoder = deit_tiny_patch16_224_TransReID(num_classes=3*cfg.MODEL.STRIDE_SIZE**2)
#         self.decoder.load_param("/home/nihao/data/checkpoints/deit_tiny_distilled_patch16_224-b40b3cf7.pth")

#         self.mask_token = nn.Parameter(torch.zeros(1, 1, 3*cfg.MODEL.STRIDE_SIZE**2))
#         self.abs_pos_embed = get_sinusoid_encoding_table(self.base.patch_embed.num_patches, 3*cfg.MODEL.STRIDE_SIZE**2)
        
#         self.discriminator = TransReID(
#             img_size=cfg.INPUT.SIZE_TRAIN,
#             # num_classes=3*cfg.MODEL.STRIDE_SIZE**2,
#             # embed_dim=3*cfg.MODEL.STRIDE_SIZE**2,
#             # depth=len(self.base.blocks),
#             # num_heads=self.base.num_heads,
#             # mlp_ratio=self.base.mlp_ratio
#             # num_classes=3*cfg.MODEL.STRIDE_SIZE**2,
#             embed_dim=cfg.MODEL.DECODER.DIM,
#             depth=cfg.MODEL.DECODER.DEPTH,
#             num_heads=cfg.MODEL.DECODER.NUM_HEAD,
#             mlp_ratio=cfg.MODEL.DECODER.MLP_RATIO,
#             norm_layer=de_norm
#         )
#         self.discriminator = deit_tiny_patch16_224_TransReID()
#         self.discriminator.load_param("/home/nihao/data/checkpoints/deit_tiny_distilled_patch16_224-b40b3cf7.pth")
#         # if 
#         self.decoder_to_discriminator = nn.Linear(self.decoder.embed_dim, 3*cfg.MODEL.STRIDE_SIZE**2, bias=False)

#         logger = logging.getLogger('reid.train')
#         logger.info("Decoder: {:.2f}M; Discriminator: {:.2f}M".format(self.decoder.compute_num_params(), self.discriminator.compute_num_params()))
#         self.d_head = nn.Linear(self.discriminator.embed_dim, 1, bias=False)
#         # self.d_head = nn.Identity()

#     def forward(self, x, mask=None, vis=False):
#         if not self.training and not vis:
#             x = self.base(x) # B, N, C
#             global_feat = x[:, 0] # cls token for global feature
#             feat = self.bottleneck(global_feat)
#             return feat if self.neck_feat == 'after' else global_feat

#         ## feat extract
#         x_enc = self.base(x)
#         global_feat = x_enc[:, 0] # cls token for global feature
        
#         ## head & bnneck for supervised ReID
#         feat = self.bottleneck(global_feat)
#         cls_score = self.classifier(feat)

#         ## img reconstruct for self-supervised generalization
#         mask = mask.bool()[:, 1:]
#         patch_size = self.cfg.MODEL.STRIDE_SIZE
#         h_num = self.cfg.INPUT.SIZE_TRAIN[0] // patch_size
#         x_enc = self.encoder_to_decoder(x_enc)[:, 1:] # drop cls token
#         B, _, C = x_enc.shape
#         x_vis = x_enc[~mask].reshape(B, -1, C)
#         expand_pos_embed = self.abs_pos_embed.expand(B, -1, -1).type_as(x_enc).to(x.device).clone().detach()
#         pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
#         pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
#         x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
#         # x_inpaint = self.decoder(x_full, pos_emd_mask.shape[1])
#         x_full = rearrange(x_full, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=h_num)
#         x_inpaint = self.decoder(x_full)
#         x_inpaint = self.decoder_to_discriminator(x_inpaint)[:,1:]
#         x_inpaint = rearrange(x_inpaint, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=h_num)
#         # x_ = self.decoder_to_discriminator(x_inpaint)
#         d_fake = torch.sigmoid(self.d_head(self.discriminator(x_inpaint)[:, 0])).squeeze()
#         d_real = torch.sigmoid(self.d_head(self.discriminator(x)[:, 0])).squeeze()
#         return cls_score, global_feat, x_inpaint, d_real, d_fake


#         # # x_enc = self.encoder_to_decoder(x_enc)
#         # x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.cfg.MODEL.STRIDE_SIZE, p2=self.cfg.MODEL.STRIDE_SIZE)
#         # pos_embed = self.abs_pos_embed.to(x.device.type)
#         # x_feat = self.decoder(x_enc[:,1:]+pos_embed)
#         # # ori_feat = self.decoder(x+pos_embed)

#         # return cls_score, global_feat, x_feat,\
#         #     #  ori_feat

#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             if 'classifier' in i: # drop classifier
#                 continue
#             if 'bottleneck' in i:
#                 continue
#             self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#         print('Loading trained model from {}'.format(trained_path))

#     def load_param_finetune(self, model_path):
#         param_dict = torch.load(model_path)
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])
#         print('Loading pretrained model for finetuning from {}'.format(model_path))

#     def compute_num_params(self):
#         total = sum([param.nelement() for param in self.parameters()])
#         logger = logging.getLogger('reid.train')
#         logger.info("Number of parameter: %.2fM" % (total/1e6))

# class colorize_decoder(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()


# ###### solve DG problem with colorization
# class build_DG_color_vit(nn.Module):
#     def __init__(self, num_classes, cfg, factory):
#         super().__init__()
#         self.cfg = cfg
#         model_path_base = cfg.MODEL.PRETRAIN_PATH
#         path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
#         self.model_path = os.path.join(model_path_base, path)
#         self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         self.in_planes = 768

#         print('using Transformer_type: vit as a backbone')

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.num_classes = num_classes
#         norm = norm_layer[cfg.MODEL.NORM.TYPE]
#         ## backbone
#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
#             (img_size=cfg.INPUT.SIZE_TRAIN,
#             stride_size=cfg.MODEL.STRIDE_SIZE,
#             drop_path_rate=cfg.MODEL.DROP_PATH,
#             drop_rate= cfg.MODEL.DROP_OUT,
#             attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
#             norm=norm)
#         if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
#             self.in_planes = 384
#         elif cfg.MODEL.TRANSFORMER_TYPE == 'deit_tiny_patch16_224_TransReID':
#             self.in_planes = 192
#         elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch16_224_TransReID':
#             self.in_planes = 1024
#         if self.pretrain_choice == 'imagenet':
#             self.base.load_param(self.model_path)
#             print('Loading pretrained ImageNet model......from {}'.format(self.model_path))
        
#         ## head & BNNeck
#         self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
#         self.classifier.apply(weights_init_classifier)
#         self.bottleneck = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck.bias.requires_grad_(False)
#         self.bottleneck.apply(weights_init_kaiming)

#         # decoder for img reconstruction
#         de_norm = norm_layer[cfg.MODEL.DECODER.NORM]
#         self.decoder = color_vit_decoder(
#             num_classes=2*cfg.MODEL.STRIDE_SIZE**2,
#             # embed_dim=cfg.MODEL.DECODER.DIM,
#             embed_dim=2*cfg.MODEL.STRIDE_SIZE**2,
#             depth=cfg.MODEL.DECODER.DEPTH,
#             num_heads=cfg.MODEL.DECODER.NUM_HEAD,
#             mlp_ratio=cfg.MODEL.DECODER.MLP_RATIO,
#             norm_layer=de_norm
#             # num_classes=3*cfg.MODEL.STRIDE_SIZE**2,
#             # embed_dim=3*cfg.MODEL.STRIDE_SIZE**2,
#             # depth=len(self.base.blocks),
#             # num_heads=self.base.num_heads,
#             # mlp_ratio=self.base.mlp_ratio
#         )
#         if self.in_planes == self.decoder.embed_dim:
#             self.encoder_to_decoder = nn.Identity()
#         else:
#             self.encoder_to_decoder = nn.Linear(self.in_planes, self.decoder.embed_dim, bias=False)

#         self.abs_pos_embed = get_sinusoid_encoding_table(self.base.patch_embed.num_patches, self.decoder.embed_dim)

#         logger = logging.getLogger('reid.train')
#         logger.info("Decoder: {:.2f}M".format(self.decoder.compute_num_params()))

#         self.ab_head = nn.Linear(self.decoder.embed_dim, 2*cfg.MODEL.STRIDE_SIZE**2, bias=False) # Lab
#         # self.d_head = nn.Identity()

#     def forward(self, x, vis=False):
#         if not self.training and not vis:
#             x = self.base(x) # B, N, C
#             global_feat = x[:, 0] # cls token for global feature
#             feat = self.bottleneck(global_feat)
#             return feat if self.neck_feat == 'after' else global_feat

#         ## feat extract
#         x_enc = self.base(x)
#         global_feat = x_enc[:, 0] # cls token for global feature
        
#         ## head & bnneck for supervised ReID
#         feat = self.bottleneck(global_feat)
#         cls_score = self.classifier(feat)

#         ## img colorization for self-supervised generalization
#         patch_size = self.cfg.MODEL.STRIDE_SIZE
#         h_num = self.cfg.INPUT.SIZE_TRAIN[0] // patch_size
#         x_enc = self.encoder_to_decoder(x_enc)[:, 1:] # drop cls token
#         x_enc += self.abs_pos_embed.to(x.device.type)
#         ab = self.decoder(x_enc)

#         ab = rearrange(ab, 'B (h w) (p1 p2 c) -> B c (h p1) (w p2)', h=h_num, p1=patch_size, p2=patch_size)

#         return cls_score, global_feat, ab


#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         count = 0
#         for i in param_dict:
#             if 'classifier' in i: # drop classifier
#                 continue
#             if 'bottleneck' in i:
#                 continue
#             if i in self.state_dict().keys():
#                 self.state_dict()[i].copy_(param_dict[i])
#                 count += 1
#         print('Loading trained model from {}\n Load {}/{} layers'.format(trained_path, count, len(self.state_dict())))

#     def load_param_finetune(self, model_path):
#         param_dict = torch.load(model_path)
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])
#         print('Loading pretrained model for finetuning from {}'.format(model_path))

#     def compute_num_params(self):
#         total = sum([param.nelement() for param in self.parameters()])
#         logger = logging.getLogger('reid.train')
#         logger.info("Number of parameter: %.2fM" % (total/1e6))

# ###### solve DG problem with rotation prediction
# class build_DG_rotation_vit(nn.Module):
#     def __init__(self, num_classes, cfg, factory):
#         super().__init__()
#         self.cfg = cfg
#         model_path_base = cfg.MODEL.PRETRAIN_PATH
#         path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
#         self.model_path = os.path.join(model_path_base, path)
#         self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         self.in_planes = 768

#         print('using Transformer_type: vit as a backbone')

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.num_classes = num_classes
#         norm = norm_layer[cfg.MODEL.NORM.TYPE]
#         ## backbone
#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
#             (img_size=cfg.INPUT.SIZE_TRAIN,
#             stride_size=cfg.MODEL.STRIDE_SIZE,
#             drop_path_rate=cfg.MODEL.DROP_PATH,
#             drop_rate= cfg.MODEL.DROP_OUT,
#             attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
#             norm=norm)
#         if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
#             self.in_planes = 384
#         elif cfg.MODEL.TRANSFORMER_TYPE == 'deit_tiny_patch16_224_TransReID':
#             self.in_planes = 192
#         elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch16_224_TransReID':
#             self.in_planes = 1024
#         if self.pretrain_choice == 'imagenet':
#             self.base.load_param(self.model_path)
#             print('Loading pretrained ImageNet model......from {}'.format(self.model_path))
        
#         ## head & BNNeck
#         self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
#         self.classifier.apply(weights_init_classifier)
#         self.bottleneck = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck.bias.requires_grad_(False)
#         self.bottleneck.apply(weights_init_kaiming)

#         # head for rotation prediction
#         self.rotation_head = nn.Linear(self.in_planes, 2, bias=False)

#     def forward(self, x, vis=False):
#         if not self.training and not vis:
#             x = self.base(x) # B, N, C
#             global_feat = x[:, 0] # cls token for global feature
#             feat = self.bottleneck(global_feat)
#             return feat if self.neck_feat == 'after' else global_feat

#         ## feat extract
#         x_enc = self.base(x)
#         global_feat = x_enc[:, 0] # cls token for global feature
        
#         ## head & bnneck for supervised ReID
#         feat = self.bottleneck(global_feat)
#         cls_score = self.classifier(feat)

#         ## rotation prediction
#         rotation_cls = self.rotation_head(global_feat)

#         return cls_score, global_feat, rotation_cls


#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         count = 0
#         for i in param_dict:
#             if 'classifier' in i: # drop classifier
#                 continue
#             if 'bottleneck' in i:
#                 continue
#             if i in self.state_dict().keys():
#                 self.state_dict()[i].copy_(param_dict[i])
#                 count += 1
#         print('Loading trained model from {}\n Load {}/{} layers'.format(trained_path, count, len(self.state_dict())))

#     def load_param_finetune(self, model_path):
#         param_dict = torch.load(model_path)
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])
#         print('Loading pretrained model for finetuning from {}'.format(model_path))

#     def compute_num_params(self):
#         total = sum([param.nelement() for param in self.parameters()])
#         logger = logging.getLogger('reid.train')
#         logger.info("Number of parameter: %.2fM" % (total/1e6))

'''
local attention vit
'''
class build_local_attention_vit(nn.Module):
    def __init__(self, num_classes, cfg, factory, pretrain_tag='imagenet'):
        super().__init__()
        self.cfg = cfg
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        if pretrain_tag == 'lup':
            path = lup_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        else:
            path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        self.model_path = os.path.join(model_path_base, path)
        self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: local token vit as a backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
            (img_size=cfg.INPUT.SIZE_TRAIN,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate= cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            pretrain_tag=pretrain_tag,
            p_num=cfg.MODEL.PART_NUM)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        elif cfg.MODEL.TRANSFORMER_TYPE == 'deit_tiny_patch16_224_TransReID':
            self.in_planes = 192
        elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch16_224_TransReID':
            self.in_planes = 1024
        if self.pretrain_choice == 'imagenet':
            self.base.load_param(self.model_path)
            print('Loading pretrained ImageNet model......from {}'.format(self.model_path))

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        # layerwise_tokens = self.base(x) # B, N, C
        # layerwise_cls_tokens = [t[:, 0] for t in layerwise_tokens] # cls token
        # local_feat_list = layerwise_tokens[-1][:, 1: 4] # 3, 768

        # layerwise_part_tokens = [[t[:, i] for i in range(1,4)] for t in layerwise_tokens] # 12 3 768
        # feat = self.bottleneck(layerwise_cls_tokens[-1])

        # if self.training:
        #     cls_score = self.classifier(feat)
        #     return cls_score, layerwise_cls_tokens, layerwise_part_tokens
        # else:
        #     return feat if self.neck_feat == 'after' else layerwise_cls_tokens[-1]
        
        x = self.base(x) # B, N, C
        global_feat = x[:, 0] # cls token for global feature
        part_token = x[:, 1:4]

        feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat, part_token
        else:
            return feat if self.neck_feat == 'after' else global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading trained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('reid.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))        

def make_model(cfg, modelname, num_class, num_class_domain_wise=None):
    if modelname == 'vit':
        model = build_vit(num_class, cfg, __factory_T_type, num_class_domain_wise)
        print('===========building vit===========')
    elif modelname == 'local_attention_vit':
        model = build_local_attention_vit(num_class, cfg, __factory_LAT_type)
        print('===========building our local attention vit===========')
    elif modelname == 'distill_vit':
        model = build_distill_vit(num_class, cfg, __factory_T_type)
        print('===========building distill vit===========')
    elif modelname == 'prompt_vit':
        model = build_prompt_vit(num_class, cfg, __factory_PT_type)
        print('===========building prompt vit===========')
    elif modelname == 'mix_vit':
        model = build_mix_vit(num_class, cfg, __factory_PT_type)
        print('===========building mix vit===========')
    elif modelname == 'XDED_vit':
        model = build_vit(num_class, cfg, __factory_T_type)
        print('===========building XDED vit===========')
    elif modelname == 'mem_vit':
        model = build_memory_cls_vit(num_class, cfg, __factory_T_type)
    elif modelname == 'mem_tri_vit':
        model = build_mix_vit(num_class, cfg, __factory_PT_type)
        print('===========building mem tri mix vit===========')
    # elif modelname == 'mask_vit':
    #     model = build_mask_vit(num_class, cfg, __factory_T_type)
    #     print('===========building mask vit===========')
    # elif modelname == 'mae':
    #     model = build_mae(num_class, cfg, __factory_T_type)
    #     print('===========building mae===========')
    # elif modelname == 'DG_ssl_vit':
    #     model = build_DG_ssl_vit(num_class, cfg, __factory_T_type)
    #     print('===========building ssl vit===========')
    # elif modelname == 'color_vit':
    #     model = build_DG_color_vit(num_class, cfg, __factory_T_type)
    #     print('===========building color vit===========')
    # elif modelname == 'rotate_vit':
    #     model = build_DG_rotation_vit(num_class, cfg, __factory_T_type)
    #     print('===========building rotate vit===========')
    else:
        model = Backbone(modelname, num_class, cfg)
        print('===========building ResNet===========')
    ### count params
    model.compute_num_params()
    return model