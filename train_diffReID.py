import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import random
import torch
import numpy as np
import argparse
from transformers import ViTModel
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL

from config import cfg
from processor.ori_vit_processor_with_amp import ori_vit_do_train_with_amp
from processor.diffusion_reid_processor import train_diffusion_reid
from utils.logger import setup_logger
from data.build_DG_dataloader import build_reid_train_loader, build_reid_test_loader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss.build_loss import build_loss

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Training")
    parser.add_argument(
        "--config_file", default="./config/vit.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(output_dir))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # build DG train loader
    train_loader, num_domains, num_pids = build_reid_train_loader(cfg)
    cfg.defrost()
    cfg.DATASETS.NUM_DOMAINS = num_domains
    cfg.freeze()
    # build DG validate loader
    val_name = cfg.DATASETS.TEST[0]
    val_loader, num_query = build_reid_test_loader(cfg, val_name)
    num_classes = len(train_loader.dataset.pids)
    model_name = cfg.MODEL.NAME

    loss_func, center_criterion = build_loss(cfg, num_classes=num_classes)

    device = 'cuda'
    image_extractor = make_model(cfg, model_name, num_classes).to(device)
    unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="unet").to(device)
    vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="vae").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="scheduler")
    vae.requires_grad_(False)
    image_extractor.train()
    unet.train()
    optimizer, optimizer_center = make_optimizer(cfg, image_extractor, unet, center_criterion)
    models = {"image_extractor": image_extractor,
              "unet": unet, "vae": vae, "noise_scheduler": noise_scheduler}

    lr_scheduler = create_scheduler(cfg, optimizer)

    

    train_diffusion_reid(
        cfg,
        models,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        lr_scheduler,
        loss_func,
        num_query, args.local_rank,
        num_pids = num_pids,
    )