import os
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial

# ------------------------------------------------------------------#
#   【修改点1】导入正确的类名 AblationSMPUnet
# ------------------------------------------------------------------#
from nets.custom_smp_unet import AblationSMPUnet

from nets.unet_training import get_lr_scheduler, set_optimizer_lr
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    # ------------------------------------------------------------------#
    #   基础配置
    # ------------------------------------------------------------------#
    Cuda            = True
    seed            = 11
    distributed     = False
    sync_bn         = False
    fp16            = True
    num_classes     = 2
    backbone        = "resnet50"
    pretrained      = True 
    model_path      = ""
    input_shape     = [512, 512]
    
    # ------------------------------------------------------------------#
    #   【消融实验开关】在这里控制你的实验变量
    # ------------------------------------------------------------------#
    # 实验1 (Baseline): False, False
    # 实验2 (Only DEES): True, False
    # 实验3 (Only GSAG): False, True
    # 实验4 (Full Method): True, True
    # ------------------------------------------------------------------#
    USE_DEES        = True   # 是否使用瓶颈层增强
    USE_GSAG        = True   # 是否使用跳跃连接注意力
    # ------------------------------------------------------------------#

    # 训练参数
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 16  
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 16 
    Freeze_Train        = False # 建议关闭冻结训练

    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    lr_decay_type       = 'cos'
    save_period         = 25
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 25
    VOCdevkit_path      = 'VOCdevkit'
    dice_loss           = True
    focal_loss          = True
    cls_weights         = np.ones([num_classes], np.float32)
    num_workers         = 8

    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0

    # ------------------------------------------------------------------#
    #   【修改点2】实例化 AblationSMPUnet 并传入开关
    # ------------------------------------------------------------------#
    print(f"Loading AblationSMPUnet with backbone: {backbone}...")
    print(f"Experiment Settings -> DEES: {USE_DEES}, GSAG: {USE_GSAG}")
    
    model = AblationSMPUnet(
        encoder_name=backbone,
        encoder_weights="imagenet",
        num_classes=num_classes,
        use_dees=USE_DEES,  # 传入开关
        use_gsag=USE_GSAG   # 传入开关
    )
    
    model = model.train()

    # 记录 Loss (为了区分实验，可以在 log_dir 加上后缀)
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    # 自动命名 log 文件夹，方便你区分是哪组实验
    exp_name        = f"_D{int(USE_DEES)}_G{int(USE_GSAG)}" 
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str) + exp_name)
    
    loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # 读取数据集
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    
    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )

    # 训练循环
    if True:
        UnFreeze_flag = False
        batch_size = Unfreeze_batch_size 
        
        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset   = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        
        train_sampler   = None
        val_sampler     = None
        shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
        
        eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                        eval_flag=eval_flag, period=eval_period)
        
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)
        
        loss_history.writer.close()