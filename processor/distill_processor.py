import logging
import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.make_model import make_model
from processor.inf_processor import do_inference
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
# from loss.build_rb_loss import RB_LOSS
from torch.cuda import amp
import torch.distributed as dist
from data.build_DG_dataloader import build_reid_test_loader, build_reid_train_loader
from torch.utils.tensorboard import SummaryWriter

def euclidean_distance(qf, gf, reduction='mean'):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    if reduction == 'sum':
        return dist_mat.sum()
    if reduction == 'mean':
        return dist_mat.mean()
    print("No such reduction!!")

def cosine_sim(qf, gf, reduction='sum'):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot)
    dist_mat = torch.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = torch.arccos(dist_mat)
    if reduction == 'sum':
        return dist_mat.sum()
    if reduction == 'mean':
        return dist_mat.mean()
    print("No such redution way!")

def Distill_do_train(cfg,
             model,
            #  center_criterion,
             train_loader,
             val_loader,
             optimizer,
            #  optimizer_center,
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
    
    # alpha = cfg.MODEL.DISTILL.ALPHA
    # beta = cfg.MODEL.DISTILL.BETA
    Lambda = cfg.MODEL.DISTILL.LAMBDA
    num_rb = cfg.MODEL.DISTILL.NUM_SELECT_BLOCK
    distill_start_epoch = cfg.MODEL.DISTILL.START_EPOCH
    hint_loss_type = cfg.MODEL.DISTILL.LOSS_TYPE
    head_flag = cfg.MODEL.DISTILL.IF_HEAD
    ds_flag = cfg.MODEL.DISTILL.IF_DEEP_SUPERVISE
    
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    reid_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    ds_loss_meter = AverageMeter()
    kl_loss_meter = AverageMeter()
    hint_loss_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler(init_scale=512) # altered by lyk
    batch_size = cfg.SOLVER.IMS_PER_BATCH # altered by lyk

    best = 0.0
    best_index = 1
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        reid_loss_meter.reset()
        ds_loss_meter.reset()
        kl_loss_meter.reset()
        hint_loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        if cfg.MODEL.FIXED_RES_BN:
            model.base.patch_embed.eval()
            
        for n_iter, informations in enumerate(train_loader):
            img = informations['images']
            vid = informations['targets']
            target_cam = informations['camid']
            # ipath = informations['img_path']
            t_domains = informations['others']['domains']

            optimizer.zero_grad()
            # optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            t_domains = t_domains.to(device)

            model.to(device)
            with amp.autocast(enabled=True):
                scores, feats, output, output_rbs, block_idx = model(img)
                '''
                loss1: reid-specific loss
                (ID + Triplet loss)
                '''
                reid_loss = loss_fn(scores[-1], feats[-1], target)
                '''
                loss2: deep supervise loss
                (ID + Triplet loss)
                '''
                ds_loss = torch.tensor(0.).to('cuda')
                if ds_flag:
                    for i in block_idx:
                        ds_loss += loss_fn(scores[i], feats[i], target)
                    ds_loss = ds_loss / num_rb
                '''
                loss3: label distribution loss
                (KL divergence)
                '''
                kl_loss = torch.tensor(0.).to('cuda')
                # if epoch >=25:
                #     import ipdb
                #     ipdb.set_trace()
                if epoch >= distill_start_epoch and head_flag:
                    for output_rb in output_rbs:
                        if cfg.MODEL.DISTILL.HARD_DISTILL:
                            mask = (output['score'] == output['score'].max(dim=1, keepdim=True)[0])
                            predicts = torch.mul(output['score'], mask)
                            kl_loss += F.cross_entropy(output_rb['score'], predicts)
                        elif cfg.MODEL.DISTILL.DOUBLE_KL:
                            kl_loss += F.kl_div(
                                F.log_softmax(output_rb['score'] / cfg.MODEL.KL_TAO, dim=1),
                                F.log_softmax(output['score'] / cfg.MODEL.KL_TAO, dim=1),
                                reduction='sum',
                                log_target=True
                            )
                            kl_loss += F.kl_div(
                                F.log_softmax(output['score'] / cfg.MODEL.KL_TAO, dim=1),
                                F.log_softmax(output_rb['score'] / cfg.MODEL.KL_TAO, dim=1),
                                reduction='sum',
                                log_target=True
                            )
                        elif cfg.MODEL.DISTILL.KL_TAO:
                            tao = cfg.MODEL.DISTILL.KL_TAO
                            kl_loss += F.kl_div( # kl loss
                                F.log_softmax(output_rb['score'] / tao, dim=1),
                                F.log_softmax(output['score'] / tao, dim=1),
                                reduction='sum',
                                log_target=True
                            )
                        else:
                            kl_loss += F.kl_div( # kl loss with norm
                                F.log_softmax(F.normalize(output_rb['score'], dim=1), dim=1),
                                F.log_softmax(F.normalize(output['score'], dim=1), dim=1),
                                reduction='sum',
                                log_target=True
                            )
                '''
                loss4: feature hint loss
                (L2 distance or else)
                '''
                hint_loss = torch.tensor(0.).to('cuda')
                for output_rb in output_rbs:
                    if hint_loss_type == 'COS':
                        hint_loss += cosine_sim(
                            output_rb['feat'],
                            output['feat'],
                            reduction='mean'
                        )
                    elif hint_loss_type == 'EUC':
                        hint_loss += euclidean_distance(
                            F.normalize(output_rb['feat'],dim=1),
                            F.normalize(output['feat'],dim=1),
                            reduction='mean'
                        )
                    elif hint_loss_type == 'L2':
                        hint_loss += F.mse_loss(
                            F.normalize(output_rb['feat'],dim=1),
                            F.normalize(output['feat'],dim=1),
                            reduction='sum'
                        ) / batch_size
                    elif hint_loss_type == 'L1':
                        hint_loss += F.l1_loss(
                            F.normalize(output_rb['feat'],dim=1),
                            F.normalize(output['feat'],dim=1),
                            reduction='mean'
                        )
                    else:
                        raise RuntimeError("No such loss type: {}.".format(hint_loss_type))
                kl_loss = kl_loss / num_rb
                hint_loss = hint_loss / num_rb
                total_loss = \
                    reid_loss * (1 - Lambda) +\
                    ds_loss * Lambda +\
                    kl_loss * Lambda +\
                    hint_loss * Lambda
                    
            scaler.scale(total_loss).backward()

            scaler.step(optimizer)
            scaler.update()

            # if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
            #     for param in center_criterion.parameters():
            #         param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
            #     scaler.step(optimizer_center)
            #     scaler.update()
            score = scores[-1]
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            reid_loss_meter.update(reid_loss.item(), img.shape[0])
            acc_meter.update(acc, 1)
            ds_loss_meter.update(ds_loss.item(), img.shape[0])
            kl_loss_meter.update(kl_loss.item(), img.shape[0])
            hint_loss_meter.update(hint_loss.item(), img.shape[0])
            
            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] reid_loss: {:.3f}, ds_loss: {:.3f}, kl_loss: {:.3f}, hint_loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                .format(epoch, n_iter+1, len(train_loader),
                reid_loss_meter.avg, ds_loss_meter.avg, kl_loss_meter.avg, hint_loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
                tbWriter.add_scalar('train/reid_loss', reid_loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
                tbWriter.add_scalar('train/acc', acc_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
                tbWriter.add_scalar('train/ds_loss', ds_loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
                tbWriter.add_scalar('train/kl_loss', kl_loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
                tbWriter.add_scalar('train/hint_loss', hint_loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))

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

        if epoch % checkpoint_period == 0:
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


