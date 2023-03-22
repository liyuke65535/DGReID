'''
@File    :   XDED_processor.py
@Time    :   2023/03/11 11:16:51
@Author  :   liyuke 
'''
import logging
import os
import time
import torch
import torch.nn as nn
from model.make_model import make_model
from processor.inf_processor import do_inference
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from data.build_DG_dataloader import build_reid_test_loader, build_reid_train_loader
from torch.utils.tensorboard import SummaryWriter

def XDED_vit_do_train_with_amp(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank,
             patch_centers = None,
             pc_criterion= None,
             train_dir=None):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid.train")
    logger.info('start training')
    log_path = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    tb_path = os.path.join(cfg.TB_LOG_ROOT, cfg.LOG_NAME)
    tbWriter = SummaryWriter(tb_path)
    print("saving tblog to {}".format(tb_path))
    
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler(init_scale=512) # altered by lyk
    batch_size = cfg.SOLVER.IMS_PER_BATCH # altered by lyk
    num_instance = cfg.DATALOADER.NUM_INSTANCE

    best = 0.0
    best_index = 1
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        
        ##### for fixed BN test
        if cfg.MODEL.FIXED_RES_BN:
            if 'res' in cfg.MODEL.NAME or 'ibn' in cfg.MODEL.NAME:
                for name, mod in model.base.named_modules():
                    if 'bn' in name:
                        mod.eval()
                        # totally freezed BN
                        mod.weight.requires_grad_(False)
                        mod.bias.requires_grad_(False)
                print("====== freeze BNs ======")
            else:
                for name, mod in model.base.named_modules():
                    if 'norm' in name:
                        mod.eval()
                        # totally freezed LN
                        mod.weight.requires_grad_(False)
                        mod.bias.requires_grad_(False)
                print("====== freeze LNs ======")
            
        
        for n_iter, informations in enumerate(train_loader):
            img = informations['images']
            vid = informations['targets']
            target_cam = informations['camid']
            # ipath = informations['img_path']
            t_domains = informations['others']['domains']

            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            t_domains = t_domains.to(device)

            model.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img)
                # classes = score.shape[-1]
                # ####### XDED
                # ####### score: (batch_size, classes)
                # score_mean = score.reshape(batch_size//num_instance,num_instance,classes).mean(1,True)
                # score_xded = score_mean.repeat(1,num_instance,1).view(-1,classes)
                loss = loss_fn(score, feat, target)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                .format(epoch, n_iter+1, len(train_loader),
                loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
                tbWriter.add_scalar('train/loss', loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
                tbWriter.add_scalar('train/acc', acc_meter.avg, n_iter+1+(epoch-1)*len(train_loader))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(epoch, time_per_batch, cfg.SOLVER.IMS_PER_BATCH / time_per_batch))

        log_path = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                cmc, mAP = do_inference(cfg, model, val_loader, num_query)
                tbWriter.add_scalar('val/Rank@1', cmc[0], epoch)
                tbWriter.add_scalar('val/mAP', mAP, epoch)
                torch.cuda.empty_cache()
        if epoch % checkpoint_period == 0:
            if best < mAP + cmc[0]:
                best = mAP + cmc[0]
                best_index = epoch
                logger.info("=====best epoch: {}=====".format(best_index))
                if cfg.MODEL.DIST_TRAIN:
                    if dist.get_rank() == 0:
                        torch.save(model.state_dict(),
                                os.path.join(log_path, cfg.MODEL.NAME + '_best.pth'))
                else:
                    torch.save(model.state_dict(),
                            os.path.join(log_path, cfg.MODEL.NAME + '_best.pth'))
        torch.cuda.empty_cache()

    # final evaluation
    load_path = os.path.join(log_path, cfg.MODEL.NAME + '_best.pth')
    eval_model = make_model(cfg, modelname=cfg.MODEL.NAME, num_class=0, camera_num=None, view_num=None)
    eval_model.load_param(load_path)
    logger.info('load weights from best.pth')
    for testname in cfg.DATASETS.TEST:
        if 'ALL' in testname:
            testname = 'DG_' + testname.split('_')[1]
        val_loader, num_query = build_reid_test_loader(cfg, testname)
        do_inference(cfg, eval_model, val_loader, num_query)