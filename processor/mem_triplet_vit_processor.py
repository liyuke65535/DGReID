import logging
import os
import time
import torch
import torch.nn as nn
from loss.triplet_loss import euclidean_dist, hard_example_mining
from loss.triplet_loss_for_mixup import hard_example_mining_for_mixup
from model.backbones.memory import FeatureMemory
from model.make_model import make_model
from processor.inf_processor import do_inference
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from data.build_DG_dataloader import build_reid_test_loader, build_reid_train_loader
from torch.utils.tensorboard import SummaryWriter

def mem_triplet_vit_do_train_with_amp(cfg,
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
             train_dir=None,
             num_pids=None,
             memories=None,
             sour_centers=None):
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
    loss_id_meter = AverageMeter()
    loss_id_distinct_meter = AverageMeter()
    loss_tri_meter = AverageMeter()
    loss_center_meter = AverageMeter()
    loss_xded_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler(init_scale=512) # altered by lyk
    bs = cfg.SOLVER.IMS_PER_BATCH # altered by lyk
    num_ins = cfg.DATALOADER.NUM_INSTANCE
    classes = len(train_loader.dataset.pids)
    center_weight = cfg.SOLVER.CENTER_LOSS_WEIGHT


    fea_mem = [FeatureMemory(768, num_pids[i]) for i in range(len(num_pids))]


    best = 0.0
    best_index = 1
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        loss_id_meter.reset()
        loss_id_distinct_meter.reset()
        loss_tri_meter.reset()
        loss_center_meter.reset()
        loss_xded_meter.reset()
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
            ori_label = informations['ori_label']
            t_domains = informations['others']['domains']

            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            ori_label = ori_label.to(device)
            t_domains = t_domains.to(device)

            targets = torch.zeros((bs, classes)).scatter_(1, target.unsqueeze(1).data.cpu(), 1).to(device)
            model.to(device)
            with amp.autocast(enabled=True):
                score, feat, targets, score_ = model(img, targets, t_domains)
                ### id loss
                log_probs = nn.LogSoftmax(dim=1)(score)
                targets = 0.9 * targets + 0.1 / classes # label smooth
                loss_id = (- targets * log_probs).mean(0).sum()
                # loss_id = torch.tensor(0.0,device=device) ####### for test

                #### id loss for each domain
                loss_id_distinct = torch.tensor(0.0, device=device)
                # for i,s in enumerate(score_):
                #     if s is None: continue
                #     idx = torch.nonzero(t_domains==i).squeeze()
                #     log_probs = nn.LogSoftmax(1)(s)
                #     label = torch.zeros((len(idx), num_pids[i])).scatter_(1, ori_label[idx].unsqueeze(1).data.cpu(), 1).to(device)
                #     label = 0.9 * label + 0.1 / num_pids[i] # label smooth
                #     loss_id_distinct += (- label * log_probs).mean(0).sum()

                # #### M3L memory loss
                # for i in range(len(memories)):
                #     idx = torch.nonzero(t_domains==i).squeeze()
                #     if len(idx) == 0: continue
                #     s = score[idx]
                #     label = torch.zeros((len(idx), num_pids[i])).scatter_(1, ori_label[idx].unsqueeze(1).data.cpu(), 1).to(device)
                #     # label = 0.9 * label + 0.1 / num_pids[i] # label smooth
                #     loss_id_distinct += memories[i](s, label).mean()

                #### memory-based tri-hard loss
                tri_hard_loss = torch.tensor(0.0, device=device)
                for i in range(len(num_pids)):
                    idx = torch.nonzero(t_domains==i).squeeze()
                    if len(idx) == 0: continue
                    fea_mem[i] = fea_mem[i].to(device)
                    tri_hard_loss += fea_mem[i](feat[idx], ori_label[idx])
                loss_tri = tri_hard_loss
                # #### triplet loss
                # target = targets.max(1)[1] ###### for mixup
                # dist_mat = euclidean_dist(feat, feat)
                # #### for mixup
                # dist_ap, dist_an = hard_example_mining_for_mixup(dist_mat, target)
                # y = dist_an.new().resize_as_(dist_an).fill_(1)
                # loss_tri = nn.SoftMarginLoss()(dist_an - dist_ap, y)
                #### center loss
                if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                    loss_center = center_criterion(feat, target)
                else:
                    loss_center = torch.tensor(0.0, device=device)
                #### XDED loss
                if cfg.MODEL.DISTILL.DO_XDED and epoch > 5:
                    probs = nn.Softmax(dim=1)(score / 0.2) # tao
                    probs_mean = probs.reshape(bs//num_ins,num_ins,classes).mean(1,True)
                    probs_xded = probs_mean.repeat(1,num_ins,1).view(-1,classes).detach()
                    loss_xded = (- probs_xded * log_probs).mean(0).sum()
                else:
                    loss_xded = torch.tensor(0.0, device=device)

                loss = loss_id + loss_tri + loss_id_distinct +\
                    center_weight * loss_center + 1.0 * loss_xded # lam

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            #### momentum update
            for i in range(len(num_pids)):
                idx = torch.nonzero(t_domains==i).squeeze()
                if len(idx)==0: continue
                fea_mem[i].momentum_update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), bs)
            loss_id_meter.update(loss_id.item(), bs)
            loss_id_distinct_meter.update(loss_id_distinct.item(), bs)
            loss_tri_meter.update(loss_tri.item(), bs)
            loss_center_meter.update(center_weight*loss_center.item(), bs)
            loss_xded_meter.update(loss_xded.item(), bs)
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, id:{:.3f}, id_dis:{:.3f}, tri:{:.3f}, cen:{:.3f}, xded:{:.3f} Acc: {:.3f}, Base Lr: {:.2e}"
                .format(epoch, n_iter+1, len(train_loader),
                loss_meter.avg,
                loss_id_meter.avg, loss_id_distinct_meter.avg, loss_tri_meter.avg, loss_center_meter.avg, loss_xded_meter.avg, 
                acc_meter.avg, scheduler._get_lr(epoch)[0]))
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
    eval_model = make_model(cfg, modelname=cfg.MODEL.NAME, num_class=0)
    eval_model.load_param(load_path)
    logger.info('load weights from best.pth')
    for testname in cfg.DATASETS.TEST:
        if 'ALL' in testname:
            testname = 'DG_' + testname.split('_')[1]
        val_loader, num_query = build_reid_test_loader(cfg, testname)
        do_inference(cfg, eval_model, val_loader, num_query)