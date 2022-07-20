import argparse
import time
import os
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('--dataset', type=str, default="KEVAL")
parser.add_argument('--modality', type=str, choices=['RGB', 'Flow'],default="RGB")
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
parser.add_argument('--video_path', type=str, default="")
parser.add_argument('--store_name', type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="R2plus1D")
parser.add_argument('--num_segments', type=int, default=32)
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll',"bce","wbce","mse","cos"])
parser.add_argument('--feature_dim', default=768, type=int, help="the feature dimension for each frame")
parser.add_argument('--pretrain', type=str, default='r2plus1d_34_32_kinetics')
parser.add_argument('--text_pretrain', type=str, default='bert-base-uncased')
parser.add_argument('--tune_from', type=str, default=None, help='fine-tune from checkpoint')
# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=24, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_scheduler_gamma', default=0.1, type=float, help='Learning rate decay factor')
parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")
# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--root_log',type=str, default='./logs')
parser.add_argument('--root_model', type=str, default='./exps')
parser.add_argument('--cfg_file', type=str, default=None, help='slowfast based cfg files')
parser.add_argument('--vmz_tune_last_k_layer', default=4, type=int,
                    help='Fine tune last k layers, range is [0,4]')
parser.add_argument('--freeze_text_to', default=0, type=int,
                    help='Fine tune from k layers, range is [0,11]')
parser.add_argument("--optimizer", default="sgd", type=str, help="optimizer types")
parser.add_argument("--lr_scheduler",default=False, action="store_true",help="whether use step scheduler")
parser.add_argument("--plat_scheduler",default=False, action="store_true",help="whether use plat scheduler")
parser.add_argument("--video_candidates",default=1,type=int,help="the number of video segments")
parser.add_argument("--bert_pooling", default="first", type=str, help="bert pooling types")
parser.add_argument("--text_model", default="bert", type=str, help="text model backbone")
parser.add_argument("--attn",action='store_true',help='if use attn')
parser.add_argument("--consist_loss",action="store_true",help="is use consist_loss")
parser.add_argument('--temp', default=0.1, type=float, help='temp')

