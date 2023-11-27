import logging
import os
import torch
import torch.nn as nn
from data.build_DG_dataloader import build_reid_test_loader
from loss.triplet_loss import TripletLoss, euclidean_dist, hard_example_mining
from model import make_model_clipreid_add_info
from processor.inf_processor import do_inference, do_inference_multi_targets
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F

from loss.supcontrast import SupConLoss
from utils.metrics import R1_mAP_eval

def do_train_stage1(cfg,
             model,
             train_loader_stage1,
             optimizer,
             scheduler,
             local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD 

    logger = logging.getLogger("reid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    image_features = []
    labels = []
    with torch.no_grad():
        for n_iter, informations in enumerate(train_loader_stage1):
            img = informations['images'].to(device)
            target = informations['targets'].to(device)
            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image = True)
                for i, img_feat in zip(target, image_feature):
                    labels.append(i)
                    image_features.append(img_feat.cpu())
        labels_list = torch.stack(labels, dim=0).cuda() #N
        image_features_list = torch.stack(image_features, dim=0).cuda()

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, image_features

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i*batch:(i+1)* batch]
            else:
                b_list = iter_list[i*batch:num_image]
            
            target = labels_list[b_list]
            image_features = image_features_list[b_list]
            with amp.autocast(enabled=True):
                text_features = model(x=image_features, label = target, get_text = True)
            loss_i2t = xent(image_features, text_features, target, target)
            loss_t2i = xent(text_features, image_features, target, target)

            loss = loss_i2t + loss_t2i

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(train_loader_stage1),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage1.batch_sampler.batch_size / time_per_batch))
        
        output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(output_dir, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(output_dir, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
                
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))

    # val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])
    # do_inference(cfg, model, val_loader, num_query)

def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("reid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    best = 0.0
    loss_meter = AverageMeter()
    i2t_acc_meter = AverageMeter()
    i2i_acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    # train
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes-batch* (num_classes//batch)
    if left != 0 :
        i_ter = i_ter+1

    # image_features = []
    # labels = []
    # with torch.no_grad():
    #     for n_iter, informations in enumerate(train_loader_stage2):
    #         img = informations['images'].to(device)
    #         target = informations['targets'].to(device)
    #         with amp.autocast(enabled=True):
    #             image_feature = model(img, target, get_image = True)
    #             for i, img_feat in zip(target, image_feature):
    #                 labels.append(i)
    #                 image_features.append(img_feat.cpu())
    #     labels_list = torch.stack(labels, dim=0).cuda() #N
    #     image_features_list = torch.stack(image_features, dim=0).cuda()

    #     batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
    #     num_image = labels_list.shape[0]
    #     i_ter = num_image // batch
    # del labels, image_features

    # text_features = []
    # with torch.no_grad():
    #     num_image = labels_list.shape[0]
    #     iter_list = torch.randperm(num_image).to(device)
    #     for i in range(i_ter):
    #         # if i+1 != i_ter:
    #         #     l_list = torch.arange(i*batch, (i+1)* batch)
    #         # else:
    #         #     l_list = torch.arange(i*batch, num_classes)

    #         target = labels_list[i*batch: (i+1)*batch]
    #         img_feature = image_features_list[i*batch: (i+1)*batch]
    #         with amp.autocast(enabled=True):
    #             text_feature = model(x = img_feature, label = target, get_text = True)
    #         text_features.append(text_feature.cpu())
    #     text_features = torch.cat(text_features, 0).cuda()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        i2t_acc_meter.reset()
        i2i_acc_meter.reset()
        evaluator.reset()

        scheduler.step()

        model.train()
        for n_iter, informations in enumerate(train_loader_stage2):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = informations['images'].to(device)
            target = informations['targets'].to(device)
            target_cam, target_view = None, None

            with amp.autocast(enabled=True):
                score, feat, image_features, text_features = model(x = img, label = target, cam_label=target_cam, view_label=target_view)
                # i2t_score = image_features @ text_features.t()

                #### image2image id loss
                bs = img.size(0)
                classes = len(train_loader_stage2.dataset.pids)
                targets = torch.zeros([bs, classes]).scatter_(1, target.unsqueeze(1).data.cpu(), 1).to(device)
                soft_targets = 0.9 * targets + 0.1 / classes # label smooth
                if isinstance(score, list):
                    log_probs = [nn.LogSoftmax(dim=1)(s) for s in score]
                    loss_id = sum([(- soft_targets * log_prob).mean(0).sum() for log_prob in log_probs])
                else:
                    log_probs = nn.LogSoftmax(dim=1)(score)
                    loss_id = (- soft_targets * log_probs).mean(0).sum()
                # loss_id = torch.tensor(0.0,device=device) ####### for test

                #### id loss for each domain
                # for i,s in enumerate(score_):
                #     if s is None: continue
                #     idx = torch.nonzero(t_domains==i).squeeze()
                #     log_probs = nn.LogSoftmax(1)(s)
                #     label = torch.zeros((len(idx), num_pids[i])).scatter_(1, ori_label[idx].unsqueeze(1).data.cpu(), 1).to(device)
                #     label = 0.9 * label + 0.1 / num_pids[i] # label smooth
                #     loss_id_distinct += (- label * log_probs).mean(0).sum()
                # loss_id_distinct = torch.tensor(0.0, device=device)

                #### triplet loss
                if isinstance(feat, list):
                    loss_tri = sum([TripletLoss()(feats, target)[0] for feats in feat[0:]])
                else:
                    dist_mat = euclidean_dist(feat, feat)
                    dist_ap, dist_an = hard_example_mining(dist_mat, target)
                    y = dist_an.new().resize_as_(dist_an).fill_(1)
                    loss_tri = nn.SoftMarginLoss()(dist_an - dist_ap, y)
                # loss_tri = torch.tensor(0.0, device=device)

                #### image2text id loss
                # log_probs_i2t = nn.LogSoftmax(dim=1)(i2t_score)
                # loss_id_i2t = (- soft_targets * log_probs_i2t).mean(0).sum()
                loss_id_i2t = torch.tensor(0.0, device=device)

                #### image2text triplet loss
                dist_mat_i2t = euclidean_dist(image_features, text_features)
                dist_ap_i2t, dist_an_i2t = hard_example_mining(dist_mat_i2t, target)
                y = dist_an_i2t.new().resize_as_(dist_an_i2t).fill_(1)
                loss_tri_i2t = nn.SoftMarginLoss()(dist_an_i2t - dist_ap_i2t, y)

                loss = loss_id + loss_tri + loss_id_i2t + loss_tri_i2t

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            # i2t_acc = (i2t_score.max(1)[1] == target).float().mean()
            i2t_acc = 0.0
            i2i_acc = (score[0].max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            i2t_acc_meter.update(i2t_acc, 1)
            i2i_acc_meter.update(i2i_acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, i2t_Acc: {:.3f}, i2i_Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, i2t_acc_meter.avg, i2i_acc_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_sampler.batch_size / time_per_batch))

        output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(output_dir, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(output_dir, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, informations in enumerate(val_loader):
                        with torch.no_grad():
                            img = informations['images'].to(device)
                            vid = informations['targets'].to(device)
                            camids = informations['camid']
                            target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camids))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                if epoch % eval_period == 0:
                    if 'DG' in cfg.DATASETS.TEST[0]:
                        cmc, mAP = do_inference_multi_targets(cfg, model, logger)
                    else:
                        cmc, mAP = do_inference(cfg, model, val_loader, num_query)
                    torch.cuda.empty_cache()
                    if best < mAP + cmc[0]:
                        best = mAP + cmc[0]
                        best_index = epoch
                        logger.info("=====best epoch: {}=====".format(best_index))
                        if cfg.MODEL.DIST_TRAIN:
                            if dist.get_rank() == 0:
                                torch.save(model.state_dict(),
                                        os.path.join(output_dir, cfg.MODEL.NAME + '_best.pth'))
                        else:
                            torch.save(model.state_dict(),
                                    os.path.join(output_dir, cfg.MODEL.NAME + '_best.pth'))
                    torch.cuda.empty_cache()

    # final evaluation
    load_path = os.path.join(output_dir, cfg.MODEL.NAME + '_best.pth')
    eval_model = make_model_clipreid_add_info.make_model(cfg, num_class=0)
    eval_model.load_param(load_path)
    logger.info('load weights from best.pth')
    if 'DG' in cfg.DATASETS.TEST[0]:
        do_inference_multi_targets(cfg, model, logger)
    else:
        for testname in cfg.DATASETS.TEST:
            val_loader, num_query = build_reid_test_loader(cfg, testname)
            do_inference(cfg, model, val_loader, num_query)