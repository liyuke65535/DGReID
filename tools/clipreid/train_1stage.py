import os
import sys
import random
import argparse
import numpy as np

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from config import cfg
from utils.logger import setup_logger
from data.make_dataloader_clipreid import make_dataloader
from solver.scheduler_factory import create_scheduler_clipreid
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss

parser = argparse.ArgumentParser(description="ReID Training")
parser.add_argument(
    "--config_file", default="config/vit_clipreid.yml", help="path to config file", type=str
)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    ######
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    print(cfg.MODEL.DEVICE_ID)
    ######

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

    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    from model.make_model_clipreid import make_model
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=sum(num_classes))

    optimizer_1stage = make_optimizer_1stage(cfg, model)
    scheduler_1stage = create_scheduler_clipreid(optimizer_1stage, num_epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS, lr_min = cfg.SOLVER.STAGE1.LR_MIN, \
                        warmup_lr_init = cfg.SOLVER.STAGE1.WARMUP_LR_INIT, warmup_t = cfg.SOLVER.STAGE1.WARMUP_EPOCHS, noise_range = None)

    from processor.processor_clipreid_1stage import do_train

    do_train(
        cfg,
        model,
        train_loader_stage1,
        optimizer_1stage,
        scheduler_1stage,
        args.local_rank
    )