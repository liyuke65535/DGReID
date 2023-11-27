import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, LLCMData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
# from model import embed_net
from model_ViT import TransReID, vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from utils import *
from loss import OriTripletLoss, CPMLoss
# from tensorboardX import SummaryWriter
from random_erasing import RandomErasing

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from model.make_model import build_vit

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='llcm', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate, 0.00035 for adam')
# parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--optim', default='AdamW', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
# parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
# parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--img_w', default=128, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=256, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int, metavar='B', help='training batch size')
# parser.add_argument('--batch-size', default=8, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=4, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--erasing_p', default=0.0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=2, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--lambda_1', default=0.8, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--lambda_2', default=0.01, type=float, help='learning rate, 0.00035 for adam')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
data_dir = "/home/guohangyu/data/datasets/"
log_dir = "/home/liyuke/data/exp/"
if dataset == 'sysu':
    data_path = data_dir + 'SYSU-MM01/'
    # data_path = '/home/guohangyu/data/datasets/SYSU-MM01'
    log_path = log_dir + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
    pool_dim = 768
elif dataset == 'regdb':
    data_path = data_dir + 'RegDB/'
    # data_path = '/home/guohangyu/data/datasets/RegDB'
    log_path = log_dir + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal
    pool_dim = 768
elif dataset == 'llcm':
    data_path = data_dir + 'LLCM/'
    # data_path = '/home/guohangyu/data/datasets/LLCM/LLCM'
    log_path = log_dir + 'llcm_log/'
    test_mode = [1, 2]  # [1, 2]: IR to VIS; [2, 1]: VIS to IR;
    pool_dim = 768

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
# suffix = suffix + '_deen_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)
suffix = suffix + '_cmtr_stride16_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)


if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
# writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_sysu = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    # transforms.RandomGrayscale(p=0.5),
    # transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability = args.erasing_p, sl = 0.2, sh = 0.8, r1 = 0.3, mean=[0.485, 0.456, 0.406]),
])
transform_regdb = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability = args.erasing_p, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.485, 0.456, 0.406]),
])
transform_llcm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    # transforms.RandomGrayscale(p=0.5),
    # transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability = args.erasing_p, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.485, 0.456, 0.406]),
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_sysu)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_regdb)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

elif dataset == 'llcm':
    # training set
    trainset = LLCMData(data_path, args.trial, transform=transform_llcm)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
net = vit_base_patch16_224_TransReID(n_class, dataset)


# 加载预训练模型的参数
net.load_param('/home/guohangyu/data/VIReID/DEENwithTransReID_new/DEENwithTransReID/model/vit_base.pth')
net.to(device)


cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = 0 #checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss()

loader_batch = args.batch_size * args.num_pos
criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)
criterion_cpm= CPMLoss(margin=0.2)

criterion_id.to(device)
criterion_tri.to(device)
criterion_cpm.to(device)

# if args.optim == 'sgd':
#     ignored_params =   list(map(id, net.bottleneck.parameters())) \
#                      + list(map(id, net.classifier.parameters()))

#     base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

#     optimizer = optim.SGD([
#         {'params': base_params, 'lr': 0.1 * args.lr},
#         {'params': net.bottleneck.parameters(), 'lr': args.lr},
#         {'params': net.classifier.parameters(), 'lr': args.lr}],
#         weight_decay=5e-4, momentum=0.9, nesterov=True)
if args.optim == 'AdamW':
    ignored_params =   list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr}],
        weight_decay=5e-4)

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# 学习率这一部分和论文提到的不一样
def adjust_learning_rate(optimizer, epoch):
    if epoch < 15:
        lr = args.lr
    elif epoch >= 15 and epoch < 30:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr
 
    return lr


def train(epoch):

    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    cpm_loss = AverageMeter()
    ort_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

        labels = torch.cat((label1, label2), 0)

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labels = Variable(labels.cuda())
        data_time.update(time.time() - end)

        feat1, feat1_att, out1 = net(input1)
        feat2, feat2_att, out2 = net(input2)
        # print(net.training)
        # print(net)

        feat = torch.cat((feat1, feat2), 0)
        feat_att = torch.cat((feat1_att, feat2_att), 0)
        out = torch.cat((out1, out2), 0)
      
        loss_id = criterion_id(out, labels)        
        loss_tri = criterion_tri(feat, labels)
        loss = loss_id + loss_tri 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # if batch_idx % 50 == 0:
        #     print('Epoch: [{}][{}/{}] '
        #           'Loss:{train_loss.val:.3f} '
        #           'iLoss:{id_loss.val:.3f} '
        #           'TLoss:{tri_loss.val:.3f} '
        #           'CLoss:{cpm_loss.val:.3f} '
        #           'OLoss:{ort_loss.val:.3f} '.format(
        #         epoch, batch_idx, len(trainloader),
        #         train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss, cpm_loss=cpm_loss, ort_loss=ort_loss))
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Loss:{train_loss.val:.3f} '
                  'iLoss:{id_loss.val:.3f} '
                  'TLoss:{tri_loss.val:.3f} '.format(
                epoch, batch_idx, len(trainloader),
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss))

    # writer.add_scalar('total_loss', train_loss.avg, epoch)
    # writer.add_scalar('id_loss', id_loss.avg, epoch)
    # writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    # writer.add_scalar('lr', current_lr, epoch)



def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, pool_dim))
    gall_feat_att = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            # feat, feat_att = net(input, input, test_mode[0])
            feat, feat_att, out = net(input)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, pool_dim))
    # feat和feat_att的区别是后者经过了BN层，而前者没有经过BN层
    query_feat_att = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            # feat, feat_att = net(input, input, test_mode[1])
            feat, feat_att, out = net(input)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))
    distmat_all = distmat + distmat_att

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_att, query_label, gall_label)
        cmc_all, mAP_all, mINP_all  = eval_regdb(-distmat_all, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
        cmc_all, mAP_all, mINP_all = eval_sysu(-distmat_all, query_label, gall_label, query_cam, gall_cam)
    elif dataset == 'llcm':
        cmc, mAP, mINP = eval_llcm(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_llcm(-distmat_att, query_label, gall_label, query_cam, gall_cam)
        cmc_all, mAP_all, mINP_all  = eval_llcm(-distmat_all, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    # writer.add_scalar('rank1', cmc[0], epoch)
    # writer.add_scalar('mAP', mAP, epoch)
    # writer.add_scalar('mINP', mINP, epoch)
    # writer.add_scalar('rank1_att', cmc_att[0], epoch)
    # writer.add_scalar('mAP_att', mAP_att, epoch)
    # writer.add_scalar('mINP_att', mINP_att, epoch)
    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att, cmc_all, mAP_all, mINP_all


# training
print('==> Start Training...')
for epoch in range(start_epoch, 151 - start_epoch):
# for epoch in range(start_epoch, 71 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    train(epoch)

    if epoch > 0 and epoch % 2 == 0:
        print('Test Epoch: {}'.format(epoch))
    
        # testing
        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att, cmc_all, mAP_all, mINP_all = test(epoch)
        # save model
        if cmc_all[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_all[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc_all,
                'mAP': mAP_all,
                'mINP': mINP_all,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')
    
        # save model
        if epoch > 10 and epoch % args.save_epoch == 0:
            state = {
                'net': net.state_dict(),
                'cmc': cmc_all,
                'mAP': mAP_all,
                'mINP': mINP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))
    
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_all[0], cmc_all[4], cmc_all[9], cmc_all[19], mAP_all, mINP_all))
        print('Best Epoch [{}]'.format(best_epoch))