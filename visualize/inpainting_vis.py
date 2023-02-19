import random
import torch
import numpy as np
import PIL.Image as Image
from PIL import Image, ImageOps
from torchvision import transforms as T
import torch.nn.functional as F
import cv2
from einops import rearrange
import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(__file__)))
from model.make_model import make_model
from config import cfg
from data.transforms.mask_generator import RandomMaskingGenerator
import ipdb

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def add_mask_to_img(img, mask, patch_size=16):
    H, W = img.shape[0] // patch_size, img.shape[1] // patch_size

    # ipdb.set_trace()
    mask_ = ~mask + 2 ## reverse
    mask_ = mask_[1:].reshape(H, W)
    # cv2.imwrite("/home/nihao/liyuke/DGReID/visualize/mask.jpg", mask_*255)
    mask_part = cv2.resize(mask_, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)[:, :, np.newaxis]
    # cv2.imwrite("/home/nihao/liyuke/DGReID/visualize/mask_resized.jpg", mask_part*255)
    m = mask_part.repeat(3, -1)

    img_ = np.array(img)
    img_[~m.astype(bool)] = 255
    # img_ = np.multiply(img,mask_part)
    return img_

transform = T.Compose([
        T.Resize((256,128), interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

###### main
def inpaint_vis(cfg, model, base_path, out_path, epoch):
    # cfg.merge_from_file("/home/nihao/liyuke/DGReID/config/reid.yml")
    # # model = make_model(modelname='DG_ssl_vit', num_class=0, cfg=cfg)
    # # model.load_param("/home/nihao/data/exp/DG_ssl_vit/vit_base/mask50/DG_ssl_vit_60.pth")
    # model = make_model(modelname='mae', num_class=0, cfg=cfg)
    # model.load_param("/home/nihao/data/exp/mae/vit_base/ratio0.5/mae_120.pth")
    model.eval()
    # model.to('cuda')
    out_path = osp.join(out_path, 'epoch'+str(epoch))
    if not osp.exists(out_path):
        os.makedirs(out_path)
    print("save vis images to {}.".format(out_path))
    mask_generator = RandomMaskingGenerator(cfg, mask_ratio=cfg.INPUT.MASK.RATIO)
    # base_path = "/home/nihao/data/market1501/query/"
    # base_path = "/home/nihao/data/market1501/bounding_box_train/"
    img_paths = os.listdir(base_path)
    random.shuffle(img_paths)

    patch_size = cfg.MODEL.STRIDE_SIZE
    for i, pth in enumerate(img_paths[:5]):
        img = Image.open(osp.join(base_path, pth))
        # img = Image.open(pth)
        img = ImageOps.exif_transpose(img) ###########
        img = img.resize((128,256))
        np_img = np.array(img)[:, :, ::-1] # BGR -> RGB
        input_tensor = transform(img).unsqueeze(0)
        mask = mask_generator().astype(int)
        mask_img = add_mask_to_img(np_img, mask)
        # cv2.imwrite(osp.join(out_path,"mask_epoch{}_{}.jpg".format(str(epoch), pth)), mask_img)
        with torch.no_grad():
            input_tensor = input_tensor.to('cuda')
            mask_torch = torch.tensor(mask).to('cuda').unsqueeze(0)
            _,_,inpaint_part = model(input_tensor, mask_torch, True)
            # _,_,inpaint_part = model(input_tensor, mask_torch)
            # inpatient_part = 
        input_tensor = input_tensor.squeeze().permute(1,2,0).cpu()*0.5+0.5
        # images_patch = rearrange(np_img, '(h p1) (w p2) c -> (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        images_patch = rearrange(input_tensor*255, '(h p1) (w p2) c -> (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        # ipdb.set_trace()
        inpaint_part = inpaint_part*0.5 + 0.5 ####### denormalize
        # inpaint_part = rearrange(inpaint_part.squeeze(), 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        images_patch = np.array(images_patch)
        if mask.sum() != 0:
            images_patch[mask[1:].astype(bool)] = np.array(inpaint_part.squeeze().cpu())*255
        else:
            images_patch = np.array(inpaint_part.cpu())*255
        img_inpaint = rearrange(images_patch, '(h w) (p1 p2 c) -> (h p1) (w p2) c', p1=patch_size, p2=patch_size, h=16)
        # cv2.imwrite(osp.join(out_path,"ori_epoch{}_{}.jpg".format(str(epoch), pth)), np.array(input_tensor*255)[:,:,::-1])
        # cv2.imwrite(osp.join(out_path,"inpaint_epoch{}_{}.jpg".format(str(epoch), pth)), img_inpaint[:,:,::-1])
        out = np.concatenate([np.array(input_tensor*255)[:,:,::-1], img_inpaint[:,:,::-1]], axis=1)
        cv2.imwrite(osp.join(out_path,"epoch{}_{}.jpg".format(str(epoch), pth)), out)
    # print('done!')