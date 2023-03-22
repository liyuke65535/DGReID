import logging
import os
import time
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
# from loss.ce_labelSmooth import CrossEntropy
from model.make_model import make_model
from processor.SSIM_loss import SSIM
from processor.inf_processor import do_inference
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from data.build_DG_dataloader import build_reid_test_loader, build_reid_train_loader
from torch.utils.tensorboard import SummaryWriter

from visualize.colorize_vis import colorize_vis

def rotate_vit_do_train_with_amp(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank,
             patch_centers,
             pc_criterion,
             train_dir
            ):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    patch_size = cfg.MODEL.STRIDE_SIZE

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
    rotation_loss_meter = AverageMeter()
    adversarial_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler(init_scale=512)

    best = 0.0
    best_index = 1

    # rotate_loss = CrossEntropy(num_classes=3)
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        rotation_loss_meter.reset()
        adversarial_loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        
        # if epoch % 2 == 1:
        #     logger.info("training Discriminator")
        # else:
        #     logger.info("training Generator")
        for n_iter, informations in enumerate(train_loader):
            img = informations['images']
            rotated_img = informations['rotated_images']
            rotation_labels = informations['rotated_labels']
            vid = informations['targets']
            target_cam = informations['camid']
            # ipath = informations['img_path']
            t_domains = informations['others']['domains']

            optimizer.zero_grad()
            img = img.to(device)
            rotated_img = rotated_img.to(device)
            rotation_labels = rotation_labels.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            t_domains = t_domains.to(device)

            model.train()
            model.to(device)
            with amp.autocast(enabled=True):
                score, feat, angles = model(rotated_img)
                # score, feat, angles2 = model(img)
                loss_reid = loss_fn(score, feat, target)

                # rotation_loss = F.mse_loss(ab, ab_target)
                rotation_loss = F.cross_entropy(angles, rotation_labels)
                # rotation_loss = F.cross_entropy(angles2, torch.zeros(img.shape[0], dtype=torch.int64).to('cuda'))
                loss = rotation_loss + loss_reid

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            rotation_loss_meter.update(rotation_loss.item(), img.shape[0])
            # adversarial_loss_meter.update(adversarial_loss.item(), B)
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, rotate_loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                .format(epoch, n_iter+1, len(train_loader),
                loss_meter.avg, rotation_loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
                tbWriter.add_scalar('train/loss', loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
                tbWriter.add_scalar('train/color_loss', rotation_loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
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
        if epoch % max(checkpoint_period,eval_period) == 0:
            if best < mAP + cmc[0]:
                best = mAP + cmc[0]
                best_index = epoch
                logger.info("=====best epoch: {}=====".format(best_index))
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
        # if epoch % cfg.SOLVER.VIS_PERIOD == 0:
        #     colorize_vis(cfg, model, train_dir, os.path.join(log_path, "vis"), epoch)
        torch.cuda.empty_cache()

    # final evaluation
    load_path = os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(best_index))
    eval_model = make_model(cfg, modelname=cfg.MODEL.NAME, num_class=0, camera_num=None, view_num=None)
    eval_model.load_param(load_path)
    print('load weights from {}_{}.pth'.format(cfg.MODEL.NAME, best_index))
    for testname in cfg.DATASETS.TEST:
        if 'ALL' in testname:
            testname = 'DG_' + testname.split('_')[1]
        val_loader, num_query = build_reid_test_loader(cfg, testname)
        do_inference(cfg, eval_model, val_loader, num_query)
    
    # remove useless path files
    del_list = os.listdir(log_path)
    for fname in del_list:
        if '.pth' in fname:
            os.remove(os.path.join(log_path, fname))
            print('removing {}. '.format(os.path.join(log_path, fname)))
    # save final checkpoint
    print('saving final checkpoint.\nDo not interrupt the program!!!')
    torch.save(eval_model.state_dict(), os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
    print('done!')

# def do_inference(cfg,
#                  model,
#                  val_loader,
#                  num_query):
#     device = "cuda"
#     logger = logging.getLogger("reid.test")
#     logger.info("Enter inferencing")

#     evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

#     evaluator.reset()

#     if device:
#         if torch.cuda.device_count() > 1:
#             print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
#             model = nn.DataParallel(model)
#         model.to(device)

#     model.eval()
#     img_path_list = []
#     t0 = time.time()
#     for n_iter, informations in enumerate(val_loader):
#         img = informations['images']
#         pid = informations['targets']
#         camids = informations['camid']
#         imgpath = informations['img_path']
#         # domains = informations['others']['domains']
#         with torch.no_grad():
#             img = img.to(device)
#             # camids = camids.to(device)
#             feat = model(img)
#             evaluator.update((feat, pid, camids))
#             img_path_list.extend(imgpath)

#     cmc, mAP, _, _, _, _, _ = evaluator.compute()
#     logger.info("Validation Results ")
#     logger.info("mAP: {:.1%}".format(mAP))
#     for r in [1, 5, 10]:
#         logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
#     logger.info("total inference time: {:.2f}".format(time.time() - t0))
#     return cmc, mAP