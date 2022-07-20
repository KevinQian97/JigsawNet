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
import random
from torch.nn import functional as F
from torch.cuda.amp import autocast,GradScaler
from scipy.spatial.distance import cdist
from ops.calc_score import calc_score
import numpy as np
# from apex import amp
# torch.autograd.set_detect_anomaly(True)


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

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        # sd = {k:v for k, v in sd.items() if k in keys2}
        sd = {k: v for k, v in sd.items() if k in keys2}
        model_dict.update(sd)
        model.load_state_dict(model_dict)
    else:
        raise RuntimeError("Please indicate the model path for evaluation")

    cudnn.benchmark = True
    normalize = GroupNormalize(input_mean, input_std)
    # Data loading code
    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    data_lists = [args.train_list, args.val_list, args.tst_list]
    results = []
    for data_list in data_lists:
        val_loader = torch.utils.data.DataLoader(
        ZSARDataset(args.root_path, data_list, num_segments=args.num_segments,
                modality=args.modality,video_candidates=args.video_candidates,
                random_shift=False,
                if_attn = args.attn,
                video_path = args.video_path,
                transform=torchvision.transforms.Compose([
                    GroupScale(scale_size),
                    GroupCenterCrop(crop_size),
                    Stack(roll=(args.arch in ['BNInception', 'InceptionV3']),inc_dim=(args.arch in ["clip","R2plus1D-34","R2plus1D-152","X3D","IP-CSN","IR-CSN"])),
                    ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3']),inc_dim=(args.arch in ["clip","R2plus1D-34","R2plus1D-152","X3D","IP-CSN","IR-CSN"])),
                    normalize,
                ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            collate_fn=collate_fn,drop_last=True)
        acc1 = validate(val_loader, model, data_list, args)
        results.append(acc1)
    print(results)
    print("AVG ACC {}, STD {}".format(sum(results)/3),np.std(results))
    return


def validate(val_loader, model, data_list, args):
    top1 = AverageMeter()
    batch_time = AverageMeter()
    if data_list == args.train_list:
        act_mat = torch.load(os.path.join(args.root_path,"tst1_act_mat.pt"))
        for k,v in act_mat.items():
            act_mat[k] = v.cuda()
        atom_dic = torch.load(os.path.join(args.root_path,"tst1_atom_dic.pt"))
    elif data_list == args.val_list:
        act_mat = torch.load(os.path.join(args.root_path,"tst2_act_mat.pt"))
        for k,v in act_mat.items():
            act_mat[k] = v.cuda()
        atom_dic = torch.load(os.path.join(args.root_path,"tst2_atom_dic.pt"))
    else:
        act_mat = torch.load(os.path.join(args.root_path,"tst3_act_mat.pt"))
        for k,v in act_mat.items():
            act_mat[k] = v.cuda()
        atom_dic = torch.load(os.path.join(args.root_path,"tst3_atom_dic.pt"))

    for k,v in atom_dic.items():
        atom_dic[k] = v.cuda()
        atom_dic[k].requires_grad = False

    model.eval()
    model.module.init_cache(act_mat)

    end = time.time()

    with torch.no_grad():
        for i, (vids,texts,objs,obj_clf,indices,act_label,obj_label,frame_ids) in enumerate(val_loader):
    
            target_act_label = act_label.cuda()
            target_obj_label = obj_label.cuda()
            input_vid = vids.cuda()
            input_text = {key:texts[key].cuda() for key in texts}
            input_obj = {key:objs[key].cuda() for key in objs}
            input_obj_clf = {key:obj_clf[key].cuda() for key in obj_clf}
        
            output = model(input_vid,input_text,input_obj,input_obj_clf)
            
            #loss
            act_logits,obj_logits = calc_score(output,model,args,indices,atom_dic)
            scores = act_logits + obj_logits.clamp(min=0)
            # measure accuracy
            if args.loss_type == "nll":
                [prec1] = accuracy_nll(scores.data, target_act_label, topk=(1,))
                top1.update(prec1.item(), vids.size(0))
            else:
                raise RuntimeError("loss type {} not supported".format(args.loss_type))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, top1=top1))
                print(output)
                logging.info(output)

    output = ('Testing Results: Prec@1 {top1.avg:.3f}'
              .format(top1=top1))
    print(output)
    logging.info(output)
    return top1.avg


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
