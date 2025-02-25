import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import time
from torch.optim import lr_scheduler
import seaborn as sns
import pandas as pd
import argparse
import os
from dataloader import MultiLoader
from loss import *
from tqdm import tqdm
import json
from model import CoSeg
import hydra
from functools import partial
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from sam2.build_sam import build_sam2
from torchmetrics.classification import BinaryAccuracy
from monai.losses import DiceCELoss, DiceLoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks import one_hot

torch.set_num_threads(8)
# matplotlib.use('TkAgg')

def train_model(model, criterion_mask, optimizer, scheduler, num_epochs=5):
    
    since = time.time()
    
    best_model_wts = model.state_dict()

    best_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
    
            else:
                model.train(False)  

            running_loss_binary = []
            running_loss_type = []
            running_loss_mse = []
            running_loss_msge = []
            running_loss_all = []
            running_loss_sem = []

            running_cls_0 = []
            running_cls_1 = []
            running_cls_2 = []
            running_cls_3 = []
            running_cls_4 = []
            running_cls_ins = []
            running_cls_sem = []
        
            # Iterate over data
            #for inputs,labels,label_for_ce,image_id in dataloaders[phase]: 
            num_samples = 0

            for img, nuclei_tissue_map, nuclei_tissue_map_re, nuclei_binary_map, \
                    nuclei_type_map, nuclei_inst_map, nuclei_hv_map, nuclei_type_map_re, img_id in tqdm(dataloaders[phase]):      
                # wrap them in Variable
    
                img = Variable(img.cuda())
                nuclei_tissue_map = Variable(nuclei_tissue_map.cuda()).unsqueeze(1)
                nuclei_tissue_map_re = Variable(nuclei_tissue_map_re.cuda()).unsqueeze(1)
                nuclei_binary_map = Variable(nuclei_binary_map.cuda()).unsqueeze(1)
                nuclei_type_map = Variable(nuclei_type_map.cuda()).unsqueeze(1)
                nuclei_inst_map = Variable(nuclei_inst_map.cuda()).unsqueeze(1)
                nuclei_hv_map = Variable(nuclei_hv_map.cuda())
                nuclei_type_map_re = Variable(nuclei_type_map_re.cuda()).unsqueeze(1)


                if phase == 'train':
                # zero the parameter gradients
                    optimizer.zero_grad()

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred_mask_ins, pred_mask_sem = model(x=img)


                    nuclei_hv_map_pred = pred_mask_ins[:,:2,:,:]
                    nuclei_binary_map_pred = pred_mask_ins[:,2:3,:,:]

                    # nuclei_binary_map_pred = torch.sigmoid(nuclei_binary_map_pred)
                    # pred_mask_sem_prob = torch.sigmoid(pred_mask_sem)
                    
                    hv_loss1 = mse_loss(input=nuclei_hv_map_pred, target=nuclei_hv_map)
                    hv_loss2 = msge_loss(input=nuclei_hv_map_pred, target=nuclei_hv_map, focus=nuclei_binary_map, device='cuda')
                    binary_loss = dice_ce_loss(nuclei_binary_map_pred, nuclei_binary_map)
                    score_mask1 = accuracy_metric(nuclei_binary_map_pred, nuclei_binary_map)
                    loss_sem = dice_ce_loss(pred_mask_sem, nuclei_tissue_map)
                    score_mask_sem = accuracy_metric(pred_mask_sem, nuclei_tissue_map)
                    # score_mask1 = torch.nan_to_num(score_mask1)

                    loss = binary_loss + loss_sem

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred_mask_ins, pred_mask_sem = model(x=img, 
                                                             prob_ins=nuclei_binary_map_pred, 
                                                             prob_sem=pred_mask_sem)
                        

                    nuclei_hv_map_pred = pred_mask_ins[:,:2,:,:]
                    nuclei_binary_map_pred = pred_mask_ins[:,2:3,:,:]

                    # nuclei_binary_map_pred = torch.sigmoid(nuclei_binary_map_pred)
                    # pred_mask_sem_prob = torch.sigmoid(pred_mask_sem)
                    
                    hv_loss1 = mse_loss(input=nuclei_hv_map_pred, target=nuclei_hv_map)
                    hv_loss2 = msge_loss(input=nuclei_hv_map_pred, target=nuclei_hv_map, focus=nuclei_binary_map, device='cuda')
                    binary_loss = dice_ce_loss(nuclei_binary_map_pred, nuclei_binary_map)
                    score_mask1 = accuracy_metric(nuclei_binary_map_pred, nuclei_binary_map)
                    loss_sem = dice_ce_loss(pred_mask_sem, nuclei_tissue_map)
                    score_mask_sem = accuracy_metric(pred_mask_sem, nuclei_tissue_map)
                    # score_mask1 = torch.nan_to_num(score_mask1)

                    loss = loss + binary_loss + 10 * hv_loss1 + 2.5 * hv_loss2 + loss_sem

                    loss.backward()
                    optimizer.step()

                else:
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            pred_mask_ins, pred_mask_sem = model(x=img)

                        nuclei_hv_map_pred = pred_mask_ins[:,:2,:,:]
                        nuclei_binary_map_pred = pred_mask_ins[:,2:3,:,:]

                        # nuclei_binary_map_pred = torch.sigmoid(nuclei_binary_map_pred)
                        # pred_mask_sem = torch.sigmoid(pred_mask_sem)
                        
                        hv_loss1 = mse_loss(input=nuclei_hv_map_pred, target=nuclei_hv_map)
                        hv_loss2 = msge_loss(input=nuclei_hv_map_pred, target=nuclei_hv_map, focus=nuclei_binary_map, device='cuda')
                        binary_loss = dice_ce_loss(nuclei_binary_map_pred, nuclei_binary_map)
                        score_mask1 = accuracy_metric(nuclei_binary_map_pred, nuclei_binary_map)
                        loss_sem = dice_ce_loss(pred_mask_sem, nuclei_tissue_map)
                        score_mask_sem = accuracy_metric(pred_mask_sem, nuclei_tissue_map)
                        # score_mask1 = torch.nan_to_num(score_mask1)

                        loss = binary_loss + loss_sem

                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            pred_mask_ins, pred_mask_sem = model(x=img, 
                                                                 prob_ins=nuclei_binary_map_pred, 
                                                                 prob_sem=pred_mask_sem)

                        nuclei_hv_map_pred = pred_mask_ins[:,:2,:,:]
                        nuclei_binary_map_pred = pred_mask_ins[:,2:3,:,:]

                        # nuclei_binary_map_pred = torch.sigmoid(nuclei_binary_map_pred)
                        # pred_mask_sem = torch.sigmoid(pred_mask_sem)
                        
                        hv_loss1 = mse_loss(input=nuclei_hv_map_pred, target=nuclei_hv_map)
                        hv_loss2 = msge_loss(input=nuclei_hv_map_pred, target=nuclei_hv_map, focus=nuclei_binary_map, device='cuda')
                        binary_loss = dice_ce_loss(nuclei_binary_map_pred, nuclei_binary_map)
                        score_mask1 = accuracy_metric(nuclei_binary_map_pred, nuclei_binary_map)
                        loss_sem = dice_ce_loss(pred_mask_sem, nuclei_tissue_map)
                        score_mask_sem = accuracy_metric(pred_mask_sem, nuclei_tissue_map)
                        # score_mask1 = torch.nan_to_num(score_mask1)

                        loss = loss + binary_loss + 10 * hv_loss1 + 2.5 * hv_loss2 + loss_sem
                    
                # calculate loss and IoU
                running_loss_all.append(loss.item())
                running_loss_binary.append(binary_loss.item())
                running_loss_sem.append(loss_sem.item())
                running_loss_mse.append(hv_loss1.item() * 10)
                running_loss_msge.append(hv_loss2.item() * 2.5)
                # running_cls_1.append(torch.mean(score_mask1[:,0]).item())
                # running_cls_2.append(torch.mean(score_mask1[:,1]).item())
                # running_cls_3.append(torch.mean(score_mask1[:,2]).item())
                # running_cls_4.append(torch.mean(score_mask1[:,3]).item())
                running_cls_ins.append(torch.mean(score_mask1).item())
                running_cls_sem.append(torch.mean(score_mask_sem).item())
                
             
            epoch_loss = np.mean(running_loss_all)
            
            # print('{} Loss: {:.4f} Stroma: {:.4f} Blood Vessel: {:.4f} Tumor: {:.4f} Epidermis: {:.4f} Avg: {:.4f}'.format(
            #     phase, epoch_loss, np.mean(running_cls_1),
            #     np.mean(running_cls_2), np.mean(running_cls_3), np.mean(running_cls_4), np.mean(running_cls_all)))

            print('{} Loss ALL: {:.4f} Loss Ins: {:.4f} Loss MSE: {:.4f} Loss MSGE: {:.4f} Loss Sem: {:.4f} Nuclei: {:.4f} Tissue: {:.4f}'.format(
                phase, epoch_loss, np.mean(running_loss_binary), np.mean(running_loss_mse), np.mean(running_loss_msge), np.mean(running_loss_sem), np.mean(running_cls_ins), np.mean(running_cls_sem)))

            # save parameters
            if phase == 'valid':
                save_point = epoch % 5
                if (epoch_loss <= best_loss) or (save_point == 0):
                    if epoch_loss <= best_loss:
                        best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    torch.save(best_model_wts, f'outputs/sam2_coseg4_{args.dataset}_{epoch}.pth')

                scheduler.step()
                print(f"lr: {scheduler.get_last_lr()[0]}")
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='puma', help='BUSI, DDTI, TN3K, UDIAT, TNBC')
    parser.add_argument('--sam_pretrain', type=str,default='/home/***/pretrain/sam2_hiera_large.pt', 
    help='pretrain/sam_vit_b_01ec64.pth, medsam_box_best_vitb.pth, medsam_vit_b, efficient_sam_vits, mobile_sam')
    parser.add_argument('--jsonfile', type=str,default='data_split.json', help='')
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=250, help='epoches')
    parser.add_argument('--size', type=float, default=1.0, help='epoches')
    args = parser.parse_args()

    os.makedirs('outputs/', exist_ok=True)

    jsonfile1 = f'/home/***/miccai2025/nuclei/datasets/{args.dataset}/data_split.json'
    
    with open(jsonfile1, 'r') as f:
        df1 = json.load(f)
    
    val_files = df1['valid']
    train_files = df1['train']

    train_dataset = MultiLoader(args.dataset, train_files, A.Compose([
        A.Resize(256, 256),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # ToTensor()
        ], 
        additional_targets={'mask2': 'mask','mask3': 'mask','mask4': 'mask','mask5': 'mask'}))
    val_dataset = MultiLoader(args.dataset, val_files, A.Compose([
        A.Resize(256, 256),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # ToTensor()
        ],
        additional_targets={'mask2': 'mask','mask3': 'mask','mask4': 'mask','mask5': 'mask'}))
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True,drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)
    
    dataloaders = {'train':train_loader,'valid':val_loader}
    

    model_cfg = "sam2_hiera_l.yaml"
    # hydra is initialized on import of sam2, which sets the search path which can't be modified
    # so we need to clear the hydra instance
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # reinit hydra with a new search path for configs
    hydra.initialize_config_module('sam2_configs', version_base='1.2')

    model = CoSeg(build_sam2(model_cfg, args.sam_pretrain, mode='train'))
    # print(model.model.sam_mask_decoder.num_multimask_outputs)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = nn.DataParallel(model)

    model = model.cuda()

    # for n, value in model.model.sam_prompt_encoder.named_parameters():
    #     value.requires_grad = False

    for n, value in model.model.image_encoder.named_parameters():
        if (f"edge" in n) or (f"neck" in n):
            value.requires_grad = True
        else:
            value.requires_grad = False

    trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
    )

    print('Trainable Params = ' + str(trainable_params/1000**2) + 'M')

    total_params = sum(
	param.numel() for param in model.parameters()
    )

    print('Total Params = ' + str(total_params/1000**2) + 'M')

    print('Ratio = ' + str(trainable_params/total_params*100) + '%')

        
    # Loss, IoU and Optimizer
    dice_ce_loss = DiceCELoss(include_background=True, to_onehot_y=False, softmax=False, sigmoid=True) # nn.CrossEntropyLoss()
    dice_ce_loss2 = DiceCELoss(include_background=True, to_onehot_y=False, softmax=False, sigmoid=True) # nn.CrossEntropyLoss()
    mse_loss = MSELossMaps()
    msge_loss = MSGELossMaps()
    accuracy_metric = DiceEval(include_background=True, to_onehot_y=False, softmax=False, sigmoid=True)#BinaryIoU()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = args.lr)
    # optimizer = optim.Adam(model.parameters(),lr = args.lr)
    # exp_lr_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=200)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5,min_lr=1e-7)
    train_model(model, dice_ce_loss, optimizer, exp_lr_scheduler, num_epochs=args.epoch)