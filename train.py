import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import logging
import sys
import csv
import json
from torch.nn.utils import clip_grad_norm_
from ops.dataset import ZSARDataset,collate_fn
from ops.ATA import ZSAR
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy_nll, accuracy_bce
from ops.ATA import get_fine_tuning_parameters
from tensorboardX import SummaryWriter
from ops.loss import MultiCosLoss
import random
from torch.nn import functional as F
from torch.cuda.amp import autocast,GradScaler
from scipy.spatial.distance import cdist
from ops.calc_score import calc_er_contrast_score_cache
# from apex import amp
# torch.autograd.set_detect_anomaly(True)

#fix randomness to make it easier for reproduction
np.random.seed(0)
random.seed(1234567)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    global args, best_prec1, least_loss
    least_loss = 1000
    best_prec1 = -1
    args = parser.parse_args()
    if os.path.exists(os.path.join(args.root_log,"error.log")):
        os.remove(os.path.join(args.root_log,"error.log"))
    logging.basicConfig(level=logging.DEBUG,filename=os.path.join(args.root_log,"error.log"),
        filemode='a',
        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    # log_handler = open(os.path.join(args.root_log,"error.log"),"w")
    # sys.stdout = log_handler
    num_class, args.train_list, args.val_list, args.tst_list, args.root_path = dataset_config.return_dataset(args.dataset, args.modality)
    
    full_arch_name = args.arch
    args.store_name = '_'.join(
        ['ZSAR', args.dataset, full_arch_name, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    args.store_name += '_{}_{}_{}_{}_{}_{}_v{}'.format(args.bert_pooling,
    args.text_model,args.freeze_text_to,args.pretrain,
    args.vmz_tune_last_k_layer,args.loss_type,args.video_candidates)
    print('storing name: ' + args.store_name)

    check_rootfolders()
    model = ZSAR(num_class=num_class, 
            num_segments=args.num_segments,
            base_model=args.arch,
            dropout=args.dropout,
            feature_dim=args.feature_dim,
            partial_bn=not args.no_partialbn,
            pretrain=args.pretrain,
            fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
            cfg_file=args.cfg_file,
            text_pretrain=args.text_pretrain,
            video_candidates=args.video_candidates,
            bert_pooling = args.bert_pooling,
            attn = args.attn
            )

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)
    # assert model.base_model.layer4[2].conv2[0][0].weight.requires_grad
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    elif args.loss_type == "bce":
        criterion = torch.nn.BCEWithLogitsLoss().cuda()
    elif args.loss_type == "mse":
        criterion = torch.nn.MSELoss().cuda()
    elif args.loss_type == "cos":
        if args.video_candidates > 1:
            criterion = MultiCosLoss().cuda()
        else:
            criterion = torch.nn.CosineEmbeddingLoss().cuda()
    else:
        raise ValueError("Unknown loss type {}".format(args.loss_type))
    if args.optimizer=="sgd":
        if args.vmz_tune_last_k_layer < 4 or args.freeze_text_to:
            params = get_fine_tuning_parameters(model, args)
        else:
            params = model.parameters()
        optimizer = torch.optim.SGD(model.parameters(),
                            args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

    elif args.optimizer=="adam":
        if args.vmz_tune_last_k_layer < 4 or args.freeze_text_to:
            params = get_fine_tuning_parameters(model, args)
        else:
            params = model.parameters()
        optimizer = torch.optim.Adam(params, 
                                    args.lr, 
                                    weight_decay=args.weight_decay)
    else:
        raise RuntimeError("not supported optimizer {}".format(args.optimizer))


    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.lr_scheduler_gamma)
    else:
        raise RuntimeError("this version only support step scheduler")

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['metric']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.lr_scheduler:
                scheduler.load_state_dict(checkpoint["lr_scheduler"])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
            logging.info(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
            logging.error(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        # sd = {k:v for k, v in sd.items() if k in keys2}
        sd = {k: v for k, v in sd.items() if k in keys2}
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k and "projection" not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    cudnn.benchmark = True
    normalize = GroupNormalize(input_mean, input_std)
    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        ZSARDataset(args.root_path, 
            args.train_list, 
            num_segments=args.num_segments,
            modality=args.modality,
            video_candidates=args.video_candidates,
            if_attn = args.attn,
            video_path = args.video_path,
            transform=torchvision.transforms.Compose([
                train_augmentation,
                Stack(roll=(args.arch in ['BNInception', 'InceptionV3']),inc_dim=(args.arch in ["clip","R2plus1D-34","R2plus1D-152","X3D","IP-CSN","IR-CSN"])),
                ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3']),inc_dim=(args.arch in ["clip","R2plus1D-34","R2plus1D-152","X3D","IP-CSN","IR-CSN"])),
                normalize,
            ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True,collate_fn=collate_fn)  # prevent something not % n_GPU
    
    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    print(model)
    logging.info(model)
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch 
        train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)
        scheduler.step()
            
        # save model
        save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'metric': best_prec1,
                'lr_scheduler': scheduler.state_dict(),
            }, False,epoch)
                
def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    num_data = len(train_loader)
    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    
    act_mat = torch.load(os.path.join(args.root_path,"trn_act_mat.pt"))
    for k,v in act_mat.items():
        act_mat[k] = v.cuda()
    model.module.init_cache(act_mat)
    atom_dic = torch.load(os.path.join(args.root_path,"trn_atom_dic.pt"))
    # for k,v in atom_dic.items():
    #     atom_dic[k] = v.cuda()
    #     atom_dic[k].requires_grad = False

    for i, (vids,texts,objs,obj_clf,indices,act_label,obj_label,frame_ids) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        target_act_label = act_label.cuda()
        target_obj_label = obj_label.cuda()
        input_vid = vids.cuda()
        input_text = {key:texts[key].cuda() for key in texts}
        input_obj = {key:objs[key].cuda() for key in objs}
        input_obj_clf = {key:obj_clf[key].cuda() for key in obj_clf}
        
        output = model(input_vid,input_text,input_obj,input_obj_clf)
        
        #loss
        act_logits,obj_logits,er_act_logits,er_obj_logits,consist_loss = calc_er_contrast_score_cache(output,model,args,indices,atom_dic,target_act_label)


        scores = act_logits + obj_logits.clamp(min=0)
        act_loss = criterion(act_logits / args.temp,target_act_label)
        obj_loss = criterion(obj_logits / args.temp,target_act_label)
        loss = criterion(scores / args.temp,target_act_label)
        loss = loss + act_loss + obj_loss
        er_act_logits = er_act_logits / args.temp
        er_obj_logits = er_obj_logits / args.temp
        er_logits = er_act_logits + er_obj_logits
        er_act_loss = er_act_logits - torch.logsumexp(er_act_logits,1,keepdim=True)
        er_act_loss = -torch.mean(er_act_loss[target_obj_label.bool()])
        er_obj_loss = er_obj_logits - torch.logsumexp(er_obj_logits,1,keepdim=True)
        er_obj_loss = -torch.mean(er_obj_loss[target_obj_label.bool()])
        er_loss = er_logits - torch.logsumexp(er_logits,1,keepdim=True)
        er_loss = -torch.mean(er_loss[target_obj_label.bool()])
        er_loss = er_loss + er_obj_loss + er_act_loss
        loss = loss + er_loss + consist_loss

        # scaler.scale(loss).backward()
        loss.backward()
        # scaler.step(optimizer)
        optimizer.step()

        # measure accuracy and record loss
        if args.loss_type == "nll":
            [prec1] = accuracy_nll(scores.data, target_act_label, topk=(1,))
            losses.update(loss.item(), vids.size(0))
            top1.update(prec1.item(), vids.size(0))
        else:
            raise RuntimeError("loss type {} not supported".format(args.loss_type))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.9f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, lr=optimizer.param_groups[-1]['lr']))
            print(output)
            logging.info(output)
            log.write(output + '\n')
            log.flush()
        tf_writer.add_scalar("loss/train",losses.val, i+epoch*num_data)
        tf_writer.add_scalar('acc/train_top1', top1.val, i+epoch*num_data)
        tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def save_checkpoint(state, is_best,epoch):
    if is_best:
        filename = '%s/%s/best.pth.tar' % (args.root_model, args.store_name)
        torch.save(state, filename)
    filename = '%s/%s/%s.pth.tar' % (args.root_model, args.store_name,epoch)
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
