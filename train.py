import argparse
import cv2
import os
import torch
import torchvision
from engine import train_one_epoch, evaluate
import utils as utils
import pandas as pd
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from datasets.dataset import WheetDataset

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--network', default='fasterrcnn_resnet50_fpn', type=str,
                                    help='efficientdet-[d0, d1, ..]')
parser.add_argument('--save_folder', default='./weights/', type=str,
                                    help='Directory for saving checkpoint models')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
parser.add_argument('--height', default=224, type=int, metavar='N',
                            help='height of image (default: 224)')
parser.add_argument('--width', default=224, type=int, metavar='N',
                            help='width of image (default: 224)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
parser.add_argument('--num_classes', default=42, type=int,
                            help='Number of class used in model')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                            metavar='N', help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                                                dest='weight_decay')
parser.add_argument('--resume', default=None, type=str,
                                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--grad_accumulation_steps', default=1, type=int,
                    help='GPU id to use.')
args = parser.parse_args()
if(not os.path.exists(os.path.join(args.save_folder, args.network))):
    os.makedirs(os.path.join(args.save_folder, args.network))
df = pd.read_csv('./data/train.csv')
train_df , valid_df = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = WheetDataset(train_df, phase='train', height=1024, width=1024)
valid_dataset = WheetDataset(valid_df, phase='valid', height=1024, width=1024)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
    collate_fn=utils.collate_fn)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
    collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
checkpoint = []
if(args.resume is not None):
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
    params = checkpoint['parser']
    args.start_epoch = checkpoint['epoch'] + 1
    del params
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=2, pretrained_backbone=True)
if(args.resume is not None):
    model.load_state_dict(checkpoint['state_dict'])
del checkpoint
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
def get_state_dict(model):
    if type(model) == torch.nn.DataParallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    return state_dict

for epoch in trange(args.start_epoch, args.epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=50)
    # update the learning rate
    # lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, valid_loader, device=device)
    state = {
            'epoch': epoch,
            'parser': args,
            'state_dict': get_state_dict(model)
        }
    torch.save(
        state,
        os.path.join(
            args.save_folder,
            args.network,
            "{}_{}.pth".format(args.network, epoch)))
