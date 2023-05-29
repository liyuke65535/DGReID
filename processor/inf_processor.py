import logging
import time

import torch
import torch.nn as nn

from utils.metrics import R1_mAP_eval

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("reid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    print(1)
    torch.cuda.synchronize()
    print(2)
    t0 = time.time()        

    for n_iter, informations in enumerate(val_loader):
        img = informations['images']
        pid = informations['targets']
        camids = informations['camid']
        imgpath = informations['img_path']
        # domains = informations['others']['domains']
        with torch.no_grad():
            img = img.to(device)
            # camids = camids.to(device)
            feat = model(img)
            evaluator.update((feat, pid, camids))
            img_path_list.extend(imgpath)

    total_f_time = time.time() - t0
    single_f_time = total_f_time / (len(val_loader) * img.shape[0])
    num_imgs_per_sec = (len(val_loader) * img.shape[0]) / total_f_time
    logger.info("Total feature time: {:.2f}s".format(total_f_time))
    logger.info("single feature time: {:.5f}s".format(single_f_time))
    logger.info("number of images per sec: {:.2f}img/s".format(num_imgs_per_sec))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("total inference time: {:.2f}".format(time.time() - t0))
    return cmc, mAP