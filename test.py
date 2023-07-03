import os
from config import cfg
import argparse
from data.build_DG_dataloader import build_reid_test_loader
from model import make_model
from processor.inf_processor import do_inference
from utils.logger import setup_logger


if __name__ == "__main__":
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

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    model = make_model(cfg, cfg.MODEL.NAME, 0)
    if cfg.TEST.WEIGHT:
        model.load_param(cfg.TEST.WEIGHT)
    else:
        print("==== random param ====")

    for testname in cfg.DATASETS.TEST:
        if 'DG' in testname:
            cmc_avg, mAP_avg = [0 for i in range(50)], 0
            for split_id in range(10):
                if testname == 'DG_VIPeR':
                    split_id = 'split_{}a'.format(split_id+1)
                val_loader, num_query = build_reid_test_loader(cfg, testname, opt=split_id)
                cmc, mAP = do_inference(cfg, model, val_loader, num_query)
                cmc_avg += cmc
                mAP_avg += mAP
            cmc_avg /= 10
            mAP_avg /= 10
            logger.info("===== Avg Results for 10 splits of {} =====".format(testname))
            logger.info("mAP: {:.1%}".format(mAP_avg))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_avg[r - 1]))
        else:
            val_loader, num_query = build_reid_test_loader(cfg, testname)
            do_inference(cfg, model, val_loader, num_query)