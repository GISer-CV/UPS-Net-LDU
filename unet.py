import colorsys
import copy
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

# ---------------------------------------------------#
#   导入必要的库
# ---------------------------------------------------#
import segmentation_models_pytorch as smp          # 用于 smp.Unet 和 smp.Linknet
from nets.custom_smp_unet import AblationSMPUnet   # 用于 custom 魔改模型
from utils.utils import cvtColor, preprocess_input, resize_image, show_config

class Unet(object):
    _defaults = {
        # -------------------------------------------------------------------#
        #   【核心设置 1】 model_path
        #   必须指向你训练好的 Linknet 权重文件！
        #   千万不要用 Unet 的权重去加载 Linknet，否则会报错。
        # -------------------------------------------------------------------#
        "model_path"    : 'logs/linknet/best_epoch_weights.pth',

        # -------------------------------------------------------------------#
        #   【核心设置 2】 model_type: 选择要使用的模型架构
        #   可选值:
        #   'custom'  -> 你的魔改模型 (AblationSMPUnet, 支持 DEES/GSAG)
        #   'smp'     -> 官方标准 SMP Unet
        #   'linknet' -> 官方标准 SMP Linknet  <--- 这次选这个
        # -------------------------------------------------------------------#
        "model_type"    : "linknet", 

        # -------------------------------------------------------------------#
        #   【核心设置 3】 消融实验开关 (仅当 model_type='custom' 时生效)
        # -------------------------------------------------------------------#
        "use_dees"      : True, 
        "use_gsag"      : True, 

        # -------------------------------------------------------------------#
        #   基础配置
        # -------------------------------------------------------------------#
        "num_classes"   : 2,
        "backbone"      : "resnet50", # 确保和你训练 Linknet 时用的 backbone 一致
        "input_shape"   : [512, 512],
        
        # 0=原图混合, 1=纯黑白掩码 (推荐用1配合 dir_predict)
        "mix_type"      : 1,  
        "cuda"          : True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (255, 255, 255), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        
        self.generate()
        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   载入模型 (根据 model_type 自动选择)
    # ---------------------------------------------------#
    def generate(self, onnx=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading Model. Type: [{self.model_type}] Backbone: [{self.backbone}]")

        # ============================================
        # 分支 1: 魔改模型 (Custom)
        # ============================================
        if self.model_type == 'custom':
            print(f"Custom Settings -> DEES: {self.use_dees}, GSAG: {self.use_gsag}")
            self.net = AblationSMPUnet(
                encoder_name=self.backbone, 
                num_classes=self.num_classes, 
                encoder_weights=None, 
                use_dees=self.use_dees, 
                use_gsag=self.use_gsag  
            )
        
        # ============================================
        # 分支 2: 标准 SMP Unet
        # ============================================
        elif self.model_type == 'smp':
            print("Using Standard SMP Unet.")
            self.net = smp.Unet(
                encoder_name=self.backbone, 
                encoder_weights=None, 
                in_channels=3,
                classes=self.num_classes
            )

        # ============================================
        # 分支 3: 标准 SMP Linknet (本次新增)
        # ============================================
        elif self.model_type == 'linknet':
            print("Using Standard SMP Linknet.")
            self.net = smp.Linknet(
                encoder_name=self.backbone, 
                encoder_weights=None, 
                in_channels=3,
                classes=self.num_classes
            )
            
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Please use 'custom', 'smp', or 'linknet'.")

        # 加载权重
        print(f"Loading weights from {self.model_path}")
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        
        self.net = self.net.eval()
        print('Model loaded successfully.')

        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, count=False, name_classes=None):
        image       = cvtColor(image)
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            # 预测
            pr = self.net(images)
            
            # 维度处理
            pr = F.softmax(pr.permute(0,2,3,1), dim=-1).cpu().numpy()
            pr = pr[0]
            
            # 截取有效区域
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            
            # 还原尺寸
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            
            # 取最大类
            pr = pr.argmax(axis=-1)
        
        if count:
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        # 根据 mix_type 返回结果
        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image   = Image.fromarray(np.uint8(seg_img))
            image   = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            seg_img = np.zeros((orininal_h, orininal_w), dtype=np.uint8)
            seg_img[pr == 1] = 255
            image = Image.fromarray(seg_img)

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            image = Image.fromarray(np.uint8(seg_img))
        
        return image

    def get_FPS(self, image, test_interval):
        image       = cvtColor(image)
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            pr = self.net(images)
            pr = F.softmax(pr.permute(0,2,3,1), dim=-1).cpu().numpy()
            pr = pr[0].argmax(axis=-1)
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                pr = self.net(images)
                pr = F.softmax(pr.permute(0,2,3,1), dim=-1).cpu().numpy()
                pr = pr[0].argmax(axis=-1)
                pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                        int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_miou_png(self, image):
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            pr = self.net(images)
            pr = F.softmax(pr.permute(0,2,3,1), dim=-1).cpu().numpy()
            pr = pr[0]
            
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)
        
        image = Image.fromarray(np.uint8(pr))
        return image