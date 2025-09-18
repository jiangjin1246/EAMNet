import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *

from models.EAMNet import EAMNet
from engine import *
import os
import sys


from utils import *
from configs.config_setting import setting_config
from dataload.dataset import BaseDataSets

import warnings
warnings.filterwarnings("ignore")


def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    #resume_model = 'resume_model.pth' 
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    set_seed(config.seed)
    #gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()





    print('#----------Preparing dataset----------#')
    input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    train_dataset = BaseDataSets(base_dir=config.data_path, split="train", num=None, transform=input_transform)
    

    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    
    val_dataset = BaseDataSets(base_dir=config.data_path, split="val", transform=input_transform)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)
   
    test_dataset = BaseDataSets(base_dir=config.data_path, split="val", transform=input_transform)
    test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)




    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    model = EAMNet(num_classes=model_cfg['num_classes'], 
                               input_channels=model_cfg['input_channels'], 
                               c_list=model_cfg['c_list'], 
                               split_att=model_cfg['split_att'], 
                               bridge=model_cfg['bridge'],)
    
    model = torch.nn.DataParallel(model.cuda())


    pretrain =False
    if pretrain:
        path_to_pretrained_model = 'pretrained_model.pth'
        # 加载预训练参数
        pretrained_dict = torch.load(path_to_pretrained_model)
    
        # 获取当前模型的参数字典
        model_dict = model.module.state_dict()

        # 筛选出需要更新的参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # 更新模型字典
        model_dict.update(pretrained_dict)

        # 将更新后的字典加载到模型中
        model.module.load_state_dict(model_dict)
      


    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    max_epoch = 1
    max_miou = 0





    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        #pdb.set_trace()
        max_loss, max_epoch, loss = checkpoint['max_miou'], checkpoint['max_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, max_miou: {max_miou:.4f}, max_epoch: {max_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            scaler=scaler
        )

        loss, miou = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )


       
        if miou > max_miou:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            max_miou = miou
            max_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'max_miou': max_miou,
                'max_epoch': max_epoch,
                'loss': loss,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth')) 

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.module.load_state_dict(best_weight)
        loss = test_one_epoch(
                test_loader,
                model,
                criterion,
                logger,
                config,
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{max_epoch}-loss{max_miou:.4f}.pth')
        )      


if __name__ == '__main__':
    config = setting_config
    main(config)
