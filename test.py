# Basic module
from tqdm                  import tqdm
from models.parse_args_test import  parse_args
import scipy.io as scio
import threading

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from dataload.dataset  import TestSetLoader
from models.metric import *
from models.loss   import *
from models.load_param_data import  load_dataset, load_param, load_dataset_eva
import torch.nn as nn
import torch.backends.cudnn as cudnn

# Model
from models.EAMNet import EAMNet
import pdb
import time
from thop import profile
import platform, os
import random
import numpy as np
import  torch


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss

class Trainer(object):
    def __init__(self, args):
        set_seed(42)
        torch.cuda.empty_cache()

        # Initial
        self.args  = args
        self.ROC   = ROCMetric(1, args.ROC_thr)
        # self.PD_FA = PD_FA(1,255)
        self.PD_FA = PD_FA(1,10, args.crop_size)
        self.mIoU  = mIoU(1)
        # 计算nIOU 完全OK
        self.nIoU_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=0)
        #self.save_prefix = '_'.join([args.model, args.dataset])
        #nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = './sirst_data/sirst'
            val_img_ids = './sirst_data/sirst/idx_427/test.txt'
            with open(val_img_ids, "r") as f:
                val_img_ids= f.readlines()
                val_img_ids= [item.replace("\n", "") for item in val_img_ids]
        if args.dataset=='NUAA_SIRST':
            Mean_Value = [.485, .456, .406]
            Std_value  = [.229, .224, .225]
            #Mean_Value = [0.2518, 0.2518, 0.2519]
            #Std_value  = [0.2557, 0.2557, 0.2558]

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(Mean_Value, Std_value)])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        #model       = res_UNet(num_classes=1, input_channels=args.in_channels, block='Res_block', num_blocks= num_blocks, nb_filter=nb_filter)
        #model.apply(weights_init_xavier)
        #print("Model Initializing")
        #self.model      = model

        # DATA_Evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Checkpoint
        ckpt_path = './checkpoint/NUAA_miou_0.7716.pth'
        checkpoint        = torch.load(ckpt_path, map_location=torch.device('cpu'))
        target_image_path = './result_WS_visual_compare'+ '/' +'NUAA_visulization_result' 
        target_dir        = './result_WS_visual_compare'+ '/' +'NUAA_visulization_fuse'

        #make_visulization_dir(target_image_path, target_dir)


        self.model = EAMNet(num_classes=1, 
                               input_channels=3, 
                               c_list=[8,16,24,32,48,64], 
                               split_att='fc', 
                               bridge=True).cuda()
        
        self.model = torch.nn.DataParallel(self.model.cuda())
       
        self.model.module.load_state_dict(checkpoint)
        #pdb.set_trace()
        self.model = self.model.to('cuda')
        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()
        with torch.no_grad():
            num = 0
            all_time = 0
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                data.cuda(non_blocking=True).float(), labels.cuda(non_blocking=True).float()
                start = time.time()
                pred = self.model(data)

                end = time.time()
                all_time +=end-start
                loss = SoftIoULoss(pred, labels) #BceDiceLoss(pred, labels)#
                #pdb.set_trace()
                #save_Ori_intensity_Pred_GT(pred, labels,target_image_path, val_img_ids, i, args.suffix,args.crop_size)

                num += 1

                losses.    update(loss.item(), pred.size(0))
                self.ROC.  update(pred, labels)
                self.mIoU. update(pred, labels)
                self.nIoU_metric.update(pred, labels)
                self.PD_FA.update(pred, labels)
                _, mean_IOU = self.mIoU.get()

            FA, PD    = self.PD_FA.get(len(val_img_ids), args.crop_size)
            test_loss = losses.avg
            # # nIOU OK Good！
            nIoU = self.nIoU_metric.get()
            ture_positive_rate, false_positive_rate, recall, precision= self.ROC.get()


            print('test_loss, %.4f' % (test_loss))
            print('mean_IOU:', mean_IOU)
            print('nIOU:', nIoU)
            print('PD:',PD)
            print('FA:',FA)
            print('all_time:',all_time)
            self.best_iou = mean_IOU
            print('TPR', ture_positive_rate)
            print('FPR', false_positive_rate)

''
def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    args = parse_args()
    main(args)





