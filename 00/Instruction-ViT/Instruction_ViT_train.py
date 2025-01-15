import torch
################################
import models.Instruction_ViT
from timm.scheduler.cosine_lr import CosineLRScheduler
import warnings
warnings.filterwarnings('ignore')
import clip
import numpy as np
###1111111111111111111111111111111111
### 102flowers AgriNet C_101 Ox_pets
from data.bulild_data_dataset_AgriNet import reset_dataloader
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.utils import accuracy, AverageMeter
import argparse
import time
from logger import create_logger
import datetime
from timm.data import Mixup
import timm

def parse_option():
    parser = argparse.ArgumentParser('SimMIM fine-tuning script', add_help=False)
    #######################222222222222222222222222222222222
    parser.add_argument('--logger_out_path',default='loggers/train_other/AgriNet' ,type=str, help='path to logger')
    parser.add_argument('--model_name',default='our_mul100' ,type=str, help='model_name')
    parser.add_argument("--local_rank", default=-1,type=int)
    parser.add_argument("--SEED", default=1234,type=int)
    parser.add_argument('--max_epoch',default=20 ,type=int, help='epochs')
    parser.add_argument('--using_VPT', action='store_true',help='if using vpt method')
    args = parser.parse_args()
    return args

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        if p.grad is None:#pass the parameter of text_features
            continue
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    return total_norm

def train_one_epoch(model,epoch,loss_fn,optimizer,train_dataloader,mixup_fn):
    
    model.train()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for samples,targets in train_dataloader:
        samples=samples.cuda(non_blocking=True)
        targets=targets.cuda(non_blocking=True)
        
        if mixup_fn is not None:
            if len(targets) % 2 == 0:
                samples, targets = mixup_fn(samples, targets)
            # else:
            #     print('not mixup',dist.get_rank())
        output,output2=model(samples)
        loss1=loss_fn(output,targets)
        loss2=loss_fn(output2,targets)
        loss = loss1+loss2
        # loss = loss2
        loss.backward()

        grad_norm = get_grad_norm(model.parameters())
        optimizer.step()
        
        optimizer.zero_grad()
        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
    epoch_time = time.time() - start
    # epoch_time = 0

    if dist.get_rank() == 0:
        logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))} loss {loss_meter.avg:.5f}")
    # logger.info(f"EPOCH {epoch},{dist.get_rank()}, training takes {datetime.timedelta(seconds=int(epoch_time))}")

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def validate(model,loss_fn,val_dataloader):
    

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    with torch.no_grad():
        model.eval()
        for images,target in val_dataloader:
            images=images.cuda(non_blocking=True)
            target=target.cuda(non_blocking=True)
            _,output=model(images)
            loss=loss_fn(output,target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)
            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))
    if dist.get_rank() == 0:
        logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f} loss{loss_meter.avg:.5f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

def main(text_features,fold_name,arg,local_rank):
    #加载ViT模型参数

    model = timm.create_model('instruction_vit_base_patch16_224',pretrained=True,num_classes=text_features.shape[0])
    if arg.using_VPT:
        print('using VPT model')
        for m,para in model.named_parameters():
            # if m in ['head.weight','head.bias','prompt_proj.weight','prompt_proj.bias']:
            if m in ['head.weight','head.bias','prompt_token']:
                # print(m)
                para.requires_grad = True
            else:
                para.requires_grad = False
    mymodel=model.to(device)
    mymodel = DDP(mymodel, device_ids=[local_rank], output_device=local_rank)

    #加载优化器 损失 
    optimizer=torch.optim.Adam(mymodel.parameters(),lr=1e-4)
    lr_schedule=CosineLRScheduler(optimizer=optimizer,t_initial=10,lr_min=1e-5,warmup_t=5)
    loss_fn= torch.nn.CrossEntropyLoss()
    epochs=arg.max_epoch
    loss_fn=loss_fn.to(device)
    mixup_fn=None
    ######################
    # text_features_temp = torch.tensor(np.random.random((text_features.shape[0],768))).to(device)
    text_features_temp = torch.tensor(text_features).to(device)
    mymodel.module.reset_prompt(text_features_temp)
    best_acc,best_epoch=0,0
    for epoch in range(epochs):
        # text_features_temp = torch.tensor(text_features).to(device)
        train_dataloader = reset_dataloader(fold_name,text_features_temp,'train')
        train_dataloader.sampler.set_epoch(epoch)
        
        mixup_fn = Mixup(
                mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
                prob=0.1, switch_prob=0.5, mode='batch',
                label_smoothing=0.1, num_classes=len(fold_name))
        
        train_one_epoch(mymodel,epoch,loss_fn,optimizer,train_dataloader,mixup_fn)

        # text_features_temp = torch.tensor(text_features).to(device)
        # mymodel.module.reset_prompt(text_features_temp)
        val_dataloader = reset_dataloader(fold_name,text_features_temp,'val')
        acc1,acc5,val_loss=validate(mymodel,loss_fn,val_dataloader)
        lr_schedule.step(epoch)
        torch.cuda.empty_cache()            
        if dist.get_rank() == 0 and epoch>=10 and acc1>best_acc:#33333333333333333333333333333333333333333333
            best_acc=acc1
            best_epoch=epoch
            torch.save(mymodel.module, "saved_parameters/other_data/AgriNet/our_mul100.pt")
    if dist.get_rank() == 0:
        logger.info(f' * best Acc@1 {best_acc:.3f} best epoch{best_epoch}')

if __name__ == '__main__':
    #读标签数据，加载CLIP，提取text
    ##44444444444444444444444444444444444444444444444444444444444444444444444444444
    text_features = np.load('data/other_dataset_mix_feature/AgriNet.npy',allow_pickle=True)
    fold_name=np.load('data/other_dataset_folds_name/AgriNet.npy',allow_pickle=True)
    arg=parse_option()
    logger = create_logger(output_dir=arg.logger_out_path, dist_rank=0, name=f"{arg.model_name}")
    local_rank = arg.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    seed = arg.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.distributed.barrier()#wait for other gpu
    device = torch.device("cuda", local_rank)
    main(text_features,fold_name,arg,local_rank)