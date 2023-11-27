import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import cfg
from data.build_DG_dataloader import build_reid_test_loader
from data.make_dataloader_clipreid import make_dataloader
from processor.inf_processor import do_inference_multi_targets
from processor.inf_processor import do_inference
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Training")
    parser.add_argument(
        "--config_file", default="config/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    from model.make_model_clipreid_add_info import make_model
    model = make_model(cfg, num_class=0, camera_num=0, view_num = 0)
    model.eval()
    model.load_param(cfg.TEST.WEIGHT)

    if cfg.DATASETS.TRAIN == 'VehicleID':
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            rank_1, rank5, mAP = do_inference(cfg,
                 model,
                 val_loader,
                 num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
                all_mAP = mAP
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5
                all_mAP = all_mAP + mAP

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, mAP, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}, sum_mAP {:.1%}".format(all_rank_1.sum()/10.0, all_rank_5.sum()/10.0, all_mAP.sum()/10.0))
    else:
       if 'DG' in cfg.DATASETS.TEST:
           do_inference_multi_targets(cfg, model, logger)
       for t in cfg.DATASETS.TEST:
            val_loader, num_query = build_reid_test_loader(cfg, t)
            do_inference(cfg, model, val_loader, num_query)