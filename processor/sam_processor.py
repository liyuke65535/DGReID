import logging
import os
import time
import torch
import torch.nn as nn
from processor.inf_processor import do_inference
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
# from torch.cuda import amp
import torch.distributed as dist
from data.build_DG_dataloader import build_reid_test_loader, build_reid_train_loader
from torch.utils.tensorboard import SummaryWriter

def sam_do_train(cfg,
             model,
             train_loader,
             val_loader,
             optimizer,
             scheduler,
             loss_fn,
             num_query, local_rank):
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

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, informations in enumerate(train_loader):
            img = informations['images']
            vid = informations['targets']
            target_cam = informations['camid']
            # ipath = informations['img_path']
            t_domains = informations['others']['domains']

            optimizer.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            t_domains = t_domains.to(device)

            model.to(device)
            score, feat = model(img)

            # first forward-backward pass
            loss = loss_fn(score, feat, target)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            score, feat = model(img)
            loss_fn(score, feat, target).backward()  # make sure to do a full forward pass
            optimizer.second_step(zero_grad=True)

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)
            tbWriter.add_scalar('train/loss', loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
            tbWriter.add_scalar('train/acc', acc_meter.avg, n_iter+1+(epoch-1)*len(train_loader))

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                .format(epoch, n_iter+1, len(train_loader),
                loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(epoch, time_per_batch, cfg.SOLVER.IMS_PER_BATCH / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

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

    # final evaluation
    for testname in cfg.DATASETS.TEST:
        if 'ALL' in testname:
            testname = 'DG_' + testname.split('_')[1]
        val_loader, num_query = build_reid_test_loader(cfg, testname)
        do_inference(cfg, model, val_loader, num_query)
    
    # save final checkpoint
    torch.save(model.state_dict(), os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

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