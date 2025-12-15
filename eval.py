import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from dataloader import MultiLoader
from skimage import measure, morphology
import albumentations as A
from albumentations.pytorch import ToTensor
from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
import argparse
import time
import pandas as pd
import cv2
import os
from skimage import io, transform
from PIL import Image
import json
from tqdm import tqdm
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from model import CoSeg
from functools import partial
from scipy import ndimage as ndi
from monai.metrics import compute_hausdorff_distance, compute_percent_hausdorff_distance, HausdorffDistanceMetric
from monai.metrics import DiceMetric, ConfusionMatrixMetric, get_confusion_matrix, compute_confusion_matrix_metric, MeanIoU
from sam2.build_sam import build_sam2
from monai.networks import one_hot
import hydra
import matplotlib.pyplot as plt


def hd_score(p, y):

    tmp_hd = compute_hausdorff_distance(p, y)
    tmp_hd = torch.mean(tmp_hd)

    return tmp_hd.item()

class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return IoU

class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth) 
        
        
        return dice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='puma',type=str, help='BUSI, DDTI, TN3K, UDIAT, FUGC, ultrasound')
    parser.add_argument('--mode', default='2d',type=str, help='2d, 3d')
    parser.add_argument('--gt_path', default='mask_1024',type=str, help='mask_1024_c1, mask_1024_')
    parser.add_argument('--jsonfile', default='data_split.json',type=str, help='')
    parser.add_argument('--size', type=int, default=1024, help='epoches')
    parser.add_argument('--model',default='/home/***/pretrain/sam2_hiera_large.pt', type=str, help='/home/***/pretrain/sam-med2d_b.pth, medsam_box_best_vitb, sam_vit_b_01ec64, medsam_vit_b, sam-med2d_b')
    args = parser.parse_args()
    
    save_png = f'visual/{args.dataset}/sam2_adapter_tissue/'
    # save_png = f"outputs/"
    feature_path = f'feature/{args.dataset}/'

    os.makedirs(save_png,exist_ok=True)
    os.makedirs(feature_path, exist_ok=True)

    print(args.dataset)
    print("------------------------------------------")

    args.jsonfile = f'/home/***/miccai2025/nuclei/datasets/{args.dataset}/data_split.json'
    if args.dataset == "refuge":
        args.jsonfile = f'/home/***/datasets/REFUGE/data_split.json'

    with open(args.jsonfile, 'r') as f:
        df = json.load(f)

    test_files = df['test'] #+ df['train'] + df['valid'] ['training_set_metastatic_roi_007.png']


    test_dataset = MultiLoader(args.dataset, test_files, A.Compose([
                                        # A.Resize(args.size, args.size),
                                        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                        # ToTensor()
                                        ], additional_targets={'mask2': 'mask'}))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1)
    model_cfg = "sam2_hiera_l.yaml"
    # hydra is initialized on import of sam2, which sets the search path which can't be modified
    # so we need to clear the hydra instance
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # reinit hydra with a new search path for configs
    hydra.initialize_config_module('sam2_configs', version_base='1.2')

    model = CoSeg(build_sam2(model_cfg, args.model, mode=None))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = nn.DataParallel(model)

    total_params = sum(
	param.numel() for param in model.parameters()
    )

    print('Total Params = ' + str(total_params/1000**2) + 'M')

    model.load_state_dict(torch.load(f'outputs/sam2_coseg4_puma_60.pth'), strict=True)

    model = model.cuda()

    idx = 0
    
    TestAcc = Accuracy()
    TestPrecision = Precision()
    TestDice = DiceMetric(include_background=True)
    TestRecall = Recall()
    TestF1 = ConfusionMatrixMetric(metric_name='f1 score')
    TestIoU = MeanIoU()

    Dice_score1 = []
    Dice_score2 = []
    Dice_score3 = []
    Dice_score4 = []
    FPS_score = []
    F1_score = []
    prec_score = []
    recall_score = []
    aji_score = []
    pq_score = []
    dq_score = []
    sq_score = []
    pq_score_tissue = []
    dq_score_tissue = []
    sq_score_tissue = []
    iou_score = []


    image_ids = []
    hd_list = []
    
    since = time.time()

    model.train(False)  
    
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for img, nuclei_tissue_map, nuclei_tissue_map_re, nuclei_binary_map, \
                    nuclei_type_map, nuclei_inst_map, nuclei_hv_map, nuclei_type_map_re, img_id in tqdm(test_loader):
        # for _, img, mask, img_id in test_loader:
            # print(img_id)
            img = Variable(img).cuda()            
            nuclei_tissue_map = Variable(nuclei_tissue_map.cuda()).unsqueeze(1)
            nuclei_tissue_map_re = Variable(nuclei_tissue_map_re.cuda()).unsqueeze(1)
            nuclei_binary_map = Variable(nuclei_binary_map.cuda()).unsqueeze(1)
            nuclei_type_map = Variable(nuclei_type_map.cuda()).unsqueeze(1)
            nuclei_inst_map = Variable(nuclei_inst_map.cuda()).unsqueeze(1)
            nuclei_hv_map = Variable(nuclei_hv_map.cuda())
            nuclei_type_map_re = Variable(nuclei_type_map_re.cuda()).unsqueeze(1)

            torch.cuda.synchronize()
            start = time.time()

            mask_pred_ins, mask_pred_sem = model(x=img)

            nuclei_hv_map_pred = mask_pred_ins[:,:2:,:]
            nuclei_binary_map_pred = mask_pred_ins[:,2:3:,:]

            mask_pred_ins, mask_pred_sem = model(x=img, prob_ins=nuclei_binary_map_pred, prob_sem=mask_pred_sem)

            nuclei_hv_map_pred = mask_pred_ins[:,:2:,:]
            nuclei_binary_map_pred = mask_pred_ins[:,2:3:,:]

            nuclei_binary_map_pred = torch.sigmoid(nuclei_binary_map_pred)
            mask_pred_sem = torch.sigmoid(mask_pred_sem)

            predictions = {"nuclei_binary_map": nuclei_binary_map_pred,
                "nuclei_type_map":  nuclei_binary_map_pred,
                "hv_map": nuclei_hv_map_pred}
            gt = {"instance_map": nuclei_inst_map,
                "nuclei_binary_map":  nuclei_binary_map,
                "instance_types_nuclei": nuclei_type_map}
            
            predictions["instance_map"], predictions["instance_types"] = model.calculate_instance_map(predictions, 40)
            # print(torch.unique(predictions["instance_map"]))
            predictions["instance_types_nuclei"] = model.generate_instance_nuclei_map(predictions["instance_map"], predictions["instance_types"]).cuda()
            predictions["batch_size"] = predictions["nuclei_binary_map"].shape[0]
            predictions["regression_map"] = None
            predictions["num_nuclei_classes"] = 2
            predictions["tissue_types"] = None

            batch_metrics = model.calculate_step_metric_validation(predictions, gt)
            nuclei_binary_map_pred[nuclei_binary_map_pred >= 0.5] = 1
            nuclei_binary_map_pred[nuclei_binary_map_pred < 0.5] = 0
            mask_pred_sem[mask_pred_sem >= 0.5] = 1
            mask_pred_sem[mask_pred_sem < 0.5] = 0

            sample_image = cv2.imread(f'/home/***/miccai2025/nuclei/datasets/puma/image_1024/{img_id[0]}.png')
            sample_image = cv2.resize(sample_image, (1024, 1024))
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(4.435,4.435))
            plt.axis('off')
            plt.imshow(sample_image)
            color = np.array([148/255, 255/255, 135/255, 0.5])
            ax = plt.gca()
            h, w = mask_pred_sem.cpu().detach().numpy()[0][0].shape
            ours = mask_pred_sem.cpu().detach().numpy()[0][0].reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(ours)
            plt.savefig(f"visual/puma/instances_6/{img_id[0]}_tissue.png",dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

            plt.figure(figsize=(4.435,4.435))
            plt.axis('off')
            plt.imshow(sample_image)
            color = np.array([148/255, 255/255, 135/255, 0.5])
            ax = plt.gca()
            h, w = nuclei_tissue_map.cpu().detach().numpy()[0][0].shape
            ours = nuclei_tissue_map.cpu().detach().numpy()[0][0].reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(ours)
            plt.savefig(f"visual/puma/instances_6/{img_id[0]}_gt_tissue.png",dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

            batch_metrics_tissue, post_tissue = model.calculate_tissue_metric_validation(mask_pred_sem, nuclei_tissue_map)
            post_tissue[post_tissue > 0] = 1
            plt.figure(figsize=(4.435,4.435))
            plt.axis('off')
            plt.imshow(sample_image)
            color = np.array([148/255, 255/255, 135/255, 0.5])
            ax = plt.gca()
            h, w = post_tissue.shape
            ours = post_tissue.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(ours)
            plt.savefig(f"visual/puma/instances_6/{img_id[0]}_tissue_post.png",dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

            torch.cuda.synchronize()
            end = time.time()
            FPS_score.append(end-start)

            batch_dice = TestDice(nuclei_binary_map_pred, nuclei_binary_map)
            batch_dice2 = TestDice(mask_pred_sem, nuclei_tissue_map)
            batch_iou = TestIoU(mask_pred_sem, nuclei_tissue_map)
            batch_cm = get_confusion_matrix(nuclei_binary_map_pred, nuclei_binary_map)
            hdscore = hd_score(mask_pred_sem,nuclei_tissue_map)
            # batch_f1 = compute_confusion_matrix_metric("f1 score", batch_cm)
            
            Dice_score1.append(batch_dice[0][0].item())
            Dice_score2.append(batch_dice2[0][0].item())
            F1_score.append(batch_metrics['f1_scores'][0])
            recall_score.append(batch_metrics['recall_scores'][0])
            prec_score.append(batch_metrics['precision_scores'][0])
            iou_score.append(batch_iou[0][0].item())
            aji_score.append(batch_metrics['aji_scores'][0])
            pq_score.append(batch_metrics['pq_scores'][0])
            dq_score.append(batch_metrics['dq_scores'][0])
            sq_score.append(batch_metrics['sq_scores'][0])
            pq_score_tissue.append(batch_metrics_tissue['pq_scores'][0])
            dq_score_tissue.append(batch_metrics_tissue['dq_scores'][0])
            sq_score_tissue.append(batch_metrics_tissue['sq_scores'][0])
            if hdscore != float("inf"):
                hd_list.append(hdscore)

            model.plot_results(
                img=img,
                predictions=predictions,
                ground_truth=gt,
                img_name=img_id[0],
                outdir="/home/***/miccai2025/nuclei/medsam2/visual/puma/instances_6",
                scores=[batch_dice[0][0].item(), batch_metrics['aji_scores'], batch_metrics['pq_scores']],
            )

            img_id = list(img_id[0].split('.'))[0]
            mask_numpy = nuclei_binary_map_pred.cpu().detach().numpy()[0][0].astype(np.uint8)
            mask_numpy[mask_numpy==1] = 255
            cv2.imwrite(f'{save_png}{img_id}.png',mask_numpy)

            mask = nuclei_tissue_map.cpu().detach().numpy()[0][0]
            mask[mask==1] = 255
            cv2.imwrite(f'{save_png}{img_id}_gt.png',mask)

            # post_tissue[post_tissue > 0] = 255
            # cv2.imwrite(f'{save_png}{img_id}_post_tissue.png',post_tissue)

            image_ids.append(img_id)
            torch.cuda.empty_cache()

            # break
         
    time_elapsed = time.time() - since


    # result_dict = {'image_id':image_ids, 'miou':mIoU, 'dice':DSC}
    # result_df = pd.DataFrame(result_dict)
    # result_df.to_csv(f'{save_png}results.csv',index=False)
    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    FPS_score.pop(0)
    print('FPS: {:.2f}'.format(1.0/(sum(FPS_score)/len(FPS_score))))
    print('Dice Nuclei:',round(np.mean(Dice_score1),4),round(np.std(Dice_score1),4))
    print('F1:',round(np.mean(F1_score),4),round(np.std(F1_score),4))
    print('Precision:',round(np.mean(prec_score),4),round(np.std(prec_score),4))
    print('Recall:',round(np.mean(recall_score),4),round(np.std(recall_score),4))
    print('AJI:',round(np.mean(aji_score),4),round(np.std(aji_score),4))
    print('PQ:',round(np.mean(pq_score),4),round(np.std(pq_score),4))
    print('DQ:',round(np.mean(dq_score),4),round(np.std(dq_score),4))
    print('SQ:',round(np.mean(sq_score),4),round(np.std(sq_score),4))
    print('Dice Tissue:',round(np.mean(Dice_score2),4),round(np.std(Dice_score2),4))
    print('mIoU:',round(np.mean(iou_score),4),round(np.std(iou_score),4))
    print('HD:',round(np.mean(hd_list),4),round(np.std(hd_list),4))
    print('PQ:',round(np.mean(pq_score_tissue),4),round(np.std(pq_score_tissue),4))
    print('DQ:',round(np.mean(dq_score_tissue),4),round(np.std(dq_score_tissue),4))
    print('SQ:',round(np.mean(sq_score_tissue),4),round(np.std(sq_score_tissue),4))
    # print('Stroma Dice:',round(np.mean(Dice_score1),4),round(np.std(Dice_score1),4))
    # print('Blood Vessel Dice:',round(np.mean(Dice_score2),4),round(np.std(Dice_score2),4))
    # print('Tumor Dice:',round(np.mean(Dice_score3),4),round(np.std(Dice_score3),4))
    # print('Epidermis Dice:',round(np.mean(Dice_score4),4),round(np.std(Dice_score4),4))
