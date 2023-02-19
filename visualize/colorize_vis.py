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
from data.data_utils import read_image
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
def colorize_vis(cfg, model, base_path, out_path, epoch):
    model.eval()
    out_path = osp.join(out_path, 'epoch'+str(epoch))
    if not osp.exists(out_path):
        os.makedirs(out_path)
    print("save vis images to {}.".format(out_path))

    img_paths = os.listdir(base_path)
    random.shuffle(img_paths)

    patch_size = cfg.MODEL.STRIDE_SIZE
    for i, pth in enumerate(img_paths[:5]):
        ori_img = read_image(osp.join(base_path, pth))
        ori_img = ori_img.resize((128,256))
        gray_img = cv2.cvtColor(np.array(ori_img), cv2.COLOR_RGB2GRAY) # h,w
        gray_img = gray_img[:,:,np.newaxis].repeat(3,axis=-1) # h,w,3
        gray_img = Image.fromarray(gray_img) # to PIL.Image
        gray_input_tensor = transform(gray_img).unsqueeze(0)

        with torch.no_grad():
            gray_input_tensor = gray_input_tensor.to('cuda')
            _,_,ab = model(gray_input_tensor, vis=True)

        gray_input_tensor = gray_input_tensor.squeeze().permute(1,2,0).cpu()*0.5+0.5
        
        # ipdb.set_trace()
        ab = ab.squeeze().permute(1,2,0)
        # ab = torch.zeros(ab.size())
        ab_part = ab * 0.5 + 0.5 ####### denormalize

        images_out = np.array(gray_input_tensor*255)
        images_out_ab = np.ones(gray_input_tensor.shape)*150

        images_out[:,:,1:] = np.array(ab_part.cpu())*255
        images_out_ab[:,:,1:] = np.array(ab_part.cpu())*255
        images_out = cv2.cvtColor(np.uint8(images_out), cv2.COLOR_LAB2BGR)
        images_out_ab = cv2.cvtColor(np.uint8(images_out_ab), cv2.COLOR_LAB2BGR)

        out = np.concatenate([np.array(gray_input_tensor*255), np.array(ori_img)[:,:,::-1], images_out_ab, images_out], axis=1)
        cv2.imwrite(osp.join(out_path,"epoch{}_{}.jpg".format(str(epoch), pth)), out)

if __name__ == '__main__':
    cfg.merge_from_file("/home/nihao/liyuke/DGReID/config/reid.yml")
    model = make_model(cfg, modelname='color_vit', num_class=0).to('cuda')
    colorize_vis(cfg, model, "/home/nihao/data/market1501/bounding_box_train/", os.path.join(osp.dirname(__file__), "vis"), 0)