import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import random
import torch
import numpy as np
import argparse

from config import cfg
from processor.XDED_processor import XDED_vit_do_train_with_amp
from processor.color_vit_processor import color_vit_do_train_with_amp
from processor.distill_processor import Distill_do_train
from processor.local_attn_vit_processor import local_attention_vit_do_train_with_amp
from processor.mae_processor import mae_do_train_with_amp
from processor.mask_vit_processor import mask_vit_do_train_with_amp
from processor.mem_triplet_vit_processor import mem_triplet_vit_do_train_with_amp
from processor.memory_classifier_vit_processor_with_amp import memory_classifier_vit_do_train_with_amp
from processor.mix_vit_processor import mix_vit_do_train_with_amp
from processor.ori_vit_processor_with_amp import ori_vit_do_train_with_amp
from processor.prompt_vit_processor_with_amp import prompt_vit_do_train_with_amp
from processor.rotate_vit_processor import rotate_vit_do_train_with_amp
from processor.sam_processor import sam_do_train
from processor.ssl_vit_processor import ssl_vit_do_train_with_amp
from utils.logger import setup_logger
from data.build_DG_dataloader import build_reid_train_loader, build_reid_test_loader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss.build_loss import build_loss
import loss as Patchloss
from model.extract_features import extract_features
import collections
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致


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
        "--config_file", default="./config/reid.yml", help="path to config file", type=str
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

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # build DG train loader
    train_loader, num_domains, num_pids, center_criterion, model = build_reid_train_loader(cfg)
    cfg.defrost()
    cfg.DATASETS.NUM_DOMAINS = num_domains
    cfg.freeze()
    # build DG validate loader
    val_name = cfg.DATASETS.TEST[0]
    val_loader, num_query = build_reid_test_loader(cfg, val_name)
    num_classes = len(train_loader.dataset.pids)
    model_name = cfg.MODEL.NAME
    # model = make_model(cfg, modelname=model_name, num_class=num_classes, num_class_domain_wise=num_pids)
    # if cfg.SOLVER.RESUME:
    #     model.load_param(cfg.SOLVER.RESUME_PATH)
    if cfg.MODEL.FREEZE_PATCH_EMBED and 'resnet' not in cfg.MODEL.NAME and 'ibn' not in cfg.MODEL.NAME: # trick from moco v3
        model.base.patch_embed.proj.weight.requires_grad = False
        model.base.patch_embed.proj.bias.requires_grad = False
        print("====== freeze patch_embed for stability ======")

    # loss_func, center_criterion = build_loss(cfg, num_classes=num_classes)
    loss_func, _ = build_loss(cfg, num_classes=num_classes)
    if cfg.MODEL.SOFT_LABEL and cfg.MODEL.NAME == 'local_attention_vit':
        print("========using soft label========")

    # #### class center init
    # source_centers_all = []
    # import torch.nn.functional as F
    # for i,testname in enumerate(cfg.DATASETS.TRAIN):
    #     sour_cluster_loader, _ = build_reid_test_loader(cfg, testname, bs=256, flag_test=False)
    #     source_features, labels = extract_features(model.base, sour_cluster_loader, print_freq=20)
    #     sour_fea_dict = collections.defaultdict(list)
    #     for k in source_features.keys():
    #         sour_fea_dict[labels[k]].append(source_features[k].unsqueeze(0))

    #     source_centers = [torch.cat(sour_fea_dict[pid], 0).mean(0) for pid in sorted(sour_fea_dict.keys())]
    #     source_centers = torch.stack(source_centers, 0).cuda()  ## pid,dim
    #     # source_centers = F.normalize(source_centers, dim=1).cuda()
    #     source_centers_all.extend(source_centers)

    #     del source_centers, sour_cluster_loader, sour_fea_dict
    # source_centers_all = torch.stack(source_centers_all, dim=0)
    # logger.info(source_centers_all.shape)
    # center_criterion.centers = torch.nn.Parameter(source_centers_all)
    # logger.info("Class Centers initiation done.")


    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)
    
    
    ################## patch loss ##############################
    #centers = Patchloss.SmoothingForImage(momentum=args.mm, num=args.num)
    patch_centers = Patchloss.PatchMemory(momentum=0.1, num=1)
    pc_criterion = Patchloss.Pedal(scale=cfg.MODEL.PC_SCALE, k=cfg.MODEL.CLUSTER_K).cuda()
    ################## patch loss ##############################

    ################ Memory Loss #################
    #### memory head
    import torch.nn.functional as F
    from model.backbones.memory import MemoryClassifier
    import collections
    from model.extract_features import extract_features
    source_centers_all = []
    memories = []
    if cfg.MODEL.NAME == 'mem_vit':
        for i,testname in enumerate(cfg.DATASETS.TRAIN):
            sour_cluster_loader, _ = build_reid_test_loader(cfg, testname, bs=256, flag_test=False)
            source_features, labels = extract_features(model.base, sour_cluster_loader, print_freq=20)
            sour_fea_dict = collections.defaultdict(list)
            for k in source_features.keys():
                sour_fea_dict[labels[k]].append(source_features[k].unsqueeze(0))

            source_centers = [torch.cat(sour_fea_dict[pid], 0).mean(0) for pid in sorted(sour_fea_dict.keys())]
            source_centers = torch.stack(source_centers, 0)  ## pid,dim
            source_centers = F.normalize(source_centers, dim=1).cuda()
            source_centers_all.append(source_centers)
        
            curMemo = MemoryClassifier(768, source_centers.shape[0]).cuda()
            curMemo.features = source_centers
            curMemo.labels = torch.arange(num_pids[i]).cuda()
            memories.append(curMemo)

            del source_centers, sour_cluster_loader, sour_fea_dict
        logger.info("Memories initiation done.") 
    
    do_train_dict = {
        'local_attention_vit': local_attention_vit_do_train_with_amp,
        'mask_vit': mask_vit_do_train_with_amp,
        'mae': mae_do_train_with_amp,
        'DG_ssl_vit': ssl_vit_do_train_with_amp,
        "color_vit": color_vit_do_train_with_amp,
        "rotate_vit": rotate_vit_do_train_with_amp,
        "prompt_vit": prompt_vit_do_train_with_amp,
        'XDED_vit': XDED_vit_do_train_with_amp,
        'mix_vit': mix_vit_do_train_with_amp,
        'mem_vit': memory_classifier_vit_do_train_with_amp,
        'mem_tri_vit': mem_triplet_vit_do_train_with_amp
    }
    if cfg.MODEL.DISTILL.DO_DISTILL:
        Distill_do_train(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_func,
            num_query, args.local_rank,
        )
    elif cfg.SOLVER.OPTIMIZER_NAME == 'SAM':
        sam_do_train(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_func,
            num_query, args.local_rank,
        )
    else :
        if model_name in do_train_dict:
            do_train_dict[model_name](
                cfg,
                model,
                center_criterion,
                train_loader,
                val_loader,
                optimizer,
                optimizer_center,
                scheduler,
                loss_func,
                num_query, args.local_rank,
                patch_centers = patch_centers,
                pc_criterion = pc_criterion,
                num_pids=num_pids,
                memories=memories,
                sour_centers=source_centers_all
            )
        # elif model_name == 'local_attention_vit':
        #     local_attention_vit_do_train_with_amp(
        #         cfg,
        #         model,
        #         center_criterion,
        #         train_loader,
        #         val_loader,
        #         optimizer,
        #         optimizer_center,
        #         scheduler,
        #         loss_func,
        #         num_query, args.local_rank,
        #         patch_centers = patch_centers,
        #         pc_criterion = pc_criterion,
        #         # num_pids = num_pids,
        #         # memories=memories,
        #         # sour_centers=source_centers_all
        #     )
        else:
            ori_vit_do_train_with_amp(
                cfg,
                model,
                center_criterion,
                train_loader,
                val_loader,
                optimizer,
                optimizer_center,
                scheduler,
                loss_func,
                num_query, args.local_rank,
                patch_centers = patch_centers,
                pc_criterion = pc_criterion,
                num_pids = num_pids,
                memories=memories,
                sour_centers=source_centers_all
            )