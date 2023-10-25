import os

import torch
from data import samplers

from data.build_DG_dataloader import build_reid_test_loader, fast_batch_collator
from data.common import CommDataset
from data.datasets import DATASET_REGISTRY
from data.transforms.build import build_transforms
from utils import comm

_root = os.getenv("REID_DATASETS", "/home/liyuke/data")

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids = [b['images'] for b in batch], [b['targets'] for b in batch], [b['camid'] for b in batch]
    # imgs = torch.tensor(imgs, dtype=torch.float32)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids,

def make_sampler_clipreid(train_set, num_batch, num_instance, num_workers, mini_batch_size, drop_last=True, flag1=True, flag2=True, seed=None, train_pids=None, cfg=None):
    if cfg.DATALOADER.SAMPLER == 'single_domain':
        data_sampler = samplers.DomainIdentitySampler(train_set.img_items,
                                                      mini_batch_size, num_instance,train_pids)
    elif flag1:
        data_sampler = samplers.RandomIdentitySampler(train_set.img_items,
                                                      mini_batch_size, num_instance)
    else:
        data_sampler = samplers.DomainSuffleSampler(train_set.img_items,
                                                     num_batch, num_instance, flag2, seed, cfg)
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, drop_last)
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator
    )
    return train_loader

def make_dataloader(cfg):
    num_workers = cfg.DATALOADER.NUM_WORKERS

    train_transforms = build_transforms(cfg, is_train=True, is_fake=False)
    train_items = list()
    domain_idx = 0
    camera_all = list()

    # load datasets
    train_pids = []
    domain_names = []
    for d in cfg.DATASETS.TRAIN:
        if d == 'CUHK03_NP':
            dataset = DATASET_REGISTRY.get('CUHK03')(root=_root, cuhk03_labeled=False)
        elif d == 'PACS':
            dataset = DATASET_REGISTRY.get(d)(root=_root)
        else:
            dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL)
        if comm.is_main_process():
            dataset.show_train()
        if len(dataset.train[0]) < 4:
            for i, x in enumerate(dataset.train):
                add_info = {}  # dictionary

                if cfg.DATALOADER.CAMERA_TO_DOMAIN and len(cfg.DATASETS.TRAIN) == 1:
                    add_info['domains'] = dataset.train[i][2]
                    camera_all.append(dataset.train[i][2])
                else:
                    add_info['domains'] = int(domain_idx)
                dataset.train[i] = list(dataset.train[i])
                dataset.train[i].append(add_info)
                dataset.train[i] = tuple(dataset.train[i])
        domain_idx += 1
        domain_names.append(dataset.dataset_name)
        train_items.extend(dataset.train)
        train_pids.append(dataset.get_num_pids(dataset.train))

    train_set = CommDataset(cfg, train_items, train_transforms, relabel=True, domain_names=domain_names)

    ####### 暂时设置为不relabel
    train_set_normal = CommDataset(cfg, train_items, build_transforms(cfg, is_train=False), relabel=True, domain_names=domain_names)
    #######

    if len(cfg.DATASETS.TRAIN) == 1 and cfg.DATALOADER.CAMERA_TO_DOMAIN:
        num_domains = dataset.num_train_cams
    else:
        num_domains = len(cfg.DATASETS.TRAIN)
    cfg.defrost()
    cfg.DATASETS.NUM_DOMAINS = num_domains
    cfg.freeze()

    train_loader_stage2 = make_sampler_clipreid(
        train_set=train_set,
        num_batch=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
        num_instance=cfg.DATALOADER.NUM_INSTANCE,
        num_workers=num_workers,
        mini_batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH // comm.get_world_size(),
        drop_last=cfg.DATALOADER.DROP_LAST,
        flag1=cfg.DATALOADER.NAIVE_WAY,
        flag2=cfg.DATALOADER.DELETE_REM,
        train_pids=train_pids,
        cfg = cfg)
    
    train_loader_stage1 = torch.utils.data.DataLoader(
        train_set_normal, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=fast_batch_collator
    )
    
    val_loader, _ = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])
    
    return train_loader_stage2, train_loader_stage1, val_loader, len(dataset.query), train_pids, 0, 0