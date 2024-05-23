import torch
import torch.nn as nn
import argparse
from config import cfg
from data.build_DG_dataloader import build_reid_test_loader

parser = argparse.ArgumentParser(description="ReID Training")
parser.add_argument(
    "--config_file", default="./config/reid.yml", help="path to config file", type=str
)
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()

if args.config_file != "":
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
val_loader, num_query = build_reid_test_loader(cfg, 'Market1501_gen', flag_test=False)
for n_iter, informations in enumerate(val_loader):
    img = informations['images']
    pid = informations['targets']
    camids = informations['camid']
    imgpath = informations['img_path']
    # domains = informations['others']['domains']
    with torch.no_grad():
        img = img.to('cuda')
        # camids = camids.to(device)
        feat = model(img)