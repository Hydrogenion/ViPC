# coding=utf-8
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-C','--cat', default=None)
parser.add_argument('-L','--lr',default=1e-4)
parser.add_argument('-G','--gpu',default='4')
parser.add_argument('-B','--bz',default='1')
parser.add_argument('-E','--epoch',default='201')
parser.add_argument('-EV','--eval_epoch',default='3')
parser.add_argument('-N','--num_workers',default='3')
parser.add_argument('-R','--resume',default='')

args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

from models import ViPC, ChamferDistance, farthest_point_sample, earth_mover_distance
from torch.utils.data import DataLoader
from utils.dataloader import *
import torch
import torch.nn as nn
from utils import meter
import torch.autograd as autograd
from torch.autograd import Variable
import time 
from tensorboardX import SummaryWriter
import kaolin as kal 
from tqdm import tqdm

if args.cat!=None:
    CLASS = args.cat
else:
    CLASS = 'plane'

MODEL = f'ViPC'
FLAG = 'train'
DEVICE = 'cuda:0'
VERSION = '1.0'
LR = float(args.lr)
BATCH_SIZE = int(args.bz)
MAX_EPOCH = int(args.epoch)
NUM_WORKERS = int(args.num_workers)
EVAL_EPOCH = int(args.eval_epoch)
RESUME = bool(args.resume)

TIME_FLAG = time.asctime(time.localtime(time.time()))
CKPT_RECORD_FOLDER = f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{LR}_{CLASS}_{FLAG}_{TIME_FLAG}/record'
CKPT_FILE = f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{LR}_{CLASS}_{FLAG}_{TIME_FLAG}/ckpt.pth'
CONFIG_FILE = f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{LR}_{CLASS}_{FLAG}_{TIME_FLAG}/CONFIG.txt'

losses_all = meter.AverageValueMeter()
losses_cd = meter.AverageValueMeter()
losses_cd_down = meter.AverageValueMeter()
losses_cd_level0 = meter.AverageValueMeter()
losses_allrid = meter.AverageValueMeter()

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_view, self.next_pcs, self.next_pc_parts = next(self.loader)
        except StopIteration:
            self.next_view = None
            self.next_pcs = None
            self.next_pc_parts = None
            return
        with torch.cuda.stream(self.stream):
            self.next_view = self.next_view.cuda(non_blocking=True)
            self.next_pcs = self.next_pcs.cuda(non_blocking=True)
            self.next_pc_parts = self.next_pc_parts.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        view = self.next_view
        pcs = self.next_pcs
        pc_parts = self.next_pc_parts

        self.preload()
        return view, pcs, pc_parts

def save_record(epoch, prec1, net:nn.Module):
    state_dict = net.state_dict()
    torch.save(state_dict,
               os.path.join(CKPT_RECORD_FOLDER, f'epoch{epoch}_{prec1:.4f}.pth'))

def save_ckpt(epoch, net , optimizer_all):
    ckpt = dict(
        epoch=epoch,
        model=net.state_dict(),
        optimizer_all=optimizer_all.state_dict(),
    )
    torch.save(ckpt,CKPT_FILE)

def train(epoch,criterion,criterion_emd,model,g_optimizer,train_loader,board_writer):
    loss = None
    prefetcher = data_prefetcher(train_loader)
    views,pcs,pc_parts = prefetcher.next()
    iteration = 0
    pbar = tqdm(total=len(train_loader))
    while views is not None:

        iteration += 1
        views = views.to(device=DEVICE)
        pcs = pcs.to(device=DEVICE)
        batch_size = views.size(0)
        g_optimizer.zero_grad()
        fine,pc_generate,level0 = model(views,pc_parts)

        indices = farthest_point_sample(pcs,2048).unsqueeze(-1).expand(BATCH_SIZE,2048,3).to(DEVICE)
        gt = torch.gather(pcs,1,indices)

        dist1, dist2 = criterion(fine, gt)
        loss_cd = (torch.mean(dist1)) + (torch.mean(dist2))

        fine = fine.repeat(3,1,1)
        gt = gt.repeat(3,1,1)

        d = criterion_emd(fine, gt ,transpose=False)
        loss_emd = d[0] / 2 + d[1] * 2 + d[2] / 3
        loss_emd = loss_emd/3


        loss = loss_cd  + loss_emd*0.0001

        loss.backward()
        g_optimizer.step()
        losses_all.add(float(loss))  # batchsize
        losses_cd.add(float(loss_cd))


        board_writer.add_scalar('loss/loss_all', losses_all.value()[0], global_step=iteration+epoch*len(train_loader))
        board_writer.add_scalar('loss/loss_cd', losses_cd.value()[0], global_step=iteration+epoch*len(train_loader))
        board_writer.add_scalar('lr',g_optimizer.state_dict()['param_groups'][0]['lr'],global_step=iteration+epoch*len(train_loader))

        views,pcs,pc_parts = prefetcher.next()

        pbar.set_postfix(loss=losses_all.value()[0])
        pbar.update(1)

    if epoch % 1 ==0:
            save_ckpt(epoch, model, g_optimizer)
            save_record(epoch, losses_all.value()[0], model)
    pbar.close()
    return loss

def model_eval(epoch,criterion,model,test_loader):
    losses_eval_cd = meter.AverageValueMeter()
    prefetcher = data_prefetcher(test_loader)
    views,pcs,pc_parts = prefetcher.next()
    # pc_parts_t = pc_parts.permute(0, 2, 1)
    iteration = 0
    pbar = tqdm(total=len(test_loader))
    model.eval()
    while views is not None:

        iteration += 1
        views = views.to(device=DEVICE)
        pcs = pcs.to(device=DEVICE)
        batch_size = views.size(0)
        fine,pc_generate,level0 = model(views,pc_parts)

        indices = farthest_point_sample(pcs,2048).unsqueeze(-1).expand(BATCH_SIZE,2048,3).to(DEVICE)
        gt = torch.gather(pcs,1,indices)
       
        dist1, dist2 = criterion(fine, gt)
        loss_cd = (torch.mean(dist1)) + (torch.mean(dist2))

        losses_eval_cd.add(float(loss_cd))

        views,pcs,pc_parts = prefetcher.next()
        pbar.set_postfix(loss=losses_eval_cd.value()[0])
        pbar.update(1)

    pbar.close()
    return loss_cd

def main():
    print('--------------------')
    print('Training Refine Net')
    print(f'Tring Class: {CLASS}')
    print('--------------------')
    model = ViPC(CLASS,False)

    model.to(device=DEVICE)
    # G = nn.DataParallel(G)


    criterion_emd = earth_mover_distance
    # criterion_grid = GriddingLoss([128,64],[0.1,0.01])
    criterion_cd= ChamferDistance()

    
    g_optimizer = torch.optim.Adam(model.parameters(), lr=LR,weight_decay=0.0001)
    g_optimizer_sgd = torch.optim.SGD(model.parameters(),lr = LR, momentum = 0.9)
    # g_optimizer = torch.optim.Adam(model.parameters(), lr=LR)


    scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size = 20, gamma = 0.1, last_epoch=-1)

    TRAIN_DATA = ViPCDataLoader('train_list.txt',data_path='Path to ShapeNetViPC-Dataset ',status='train',view_align=False,category='plane')
    TEST_DATA = ViPCDataLoader('test_list.txt',data_path='Path to ShapeNetViPC-Dataset ',status='test',view_align=False,category='plane')



    train_loader = DataLoader(TRAIN_DATA,
                              batch_size=BATCH_SIZE ,
                              num_workers=NUM_WORKERS,
                              shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(TEST_DATA,
                              batch_size=BATCH_SIZE ,
                              num_workers=NUM_WORKERS,
                              shuffle=True,
                              drop_last=True)

    resume_epoch = 0
    board_writer = SummaryWriter(comment=f'{MODEL}_{VERSION}_{BATCH_SIZE}_{LR}_{FLAG}_{CLASS}_{TIME_FLAG}')


    if RESUME:
        ckpt_path = CKPT_FILE
        ckpt_dict = torch.load(ckpt_path)
        model.load_state_dict(ckpt_dict['model'])
        g_optimizer.load_state_dict(ckpt_dict['optimizer_all'])
        # g_optimizer.param_groups[0]['lr']= 5e-6
        resume_epoch = ckpt_dict['epoch']


    if not os.path.exists(os.path.join(CKPT_RECORD_FOLDER)):
        os.makedirs(os.path.join(CKPT_RECORD_FOLDER))

    with open(CONFIG_FILE,'w') as f:
        f.write('RESUME:'+str(RESUME)+'\n')
        f.write('FLAG:'+str(FLAG)+'\n')
        f.write('DEVICE:'+str(DEVICE)+'\n')
        f.write('LR:'+str(LR)+'\n')
        f.write('BATCH_SIZE:'+str(BATCH_SIZE)+'\n')
        f.write('MAX_EPOCH:'+str(MAX_EPOCH)+'\n')
        f.write('NUM_WORKERS:'+str(NUM_WORKERS)+'\n')
        f.write('CLASS:'+str(CLASS)+'\n')
    
    # prefetcher = data_prefetcher(train_loader)
    # data = prefetcher.next()

    for epoch in range(resume_epoch, resume_epoch+MAX_EPOCH):
        best_loss = 9999
        best_epoch = 0
        scheduler.step()
        losses=train(epoch,criterion_cd,criterion_emd,model,g_optimizer,train_loader,board_writer)
        if epoch % EVAL_EPOCH ==0 and epoch!=0:
            loss = model_eval(epoch,criterion_cd,model,test_loader)
            if loss<best_loss:
                best_loss = loss
                best_epoch = epoch
            print(best_epoch,' ',best_loss)
        print('****************************')   
        print(best_epoch,' ',best_loss)
        print('****************************') 

    print('Train Finished!')

if __name__ == '__main__':
    main()