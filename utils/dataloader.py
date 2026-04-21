import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import albumentations as A  # 引入强大的数据增强库

from utils.utils import cvtColor, preprocess_input

class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(UnetDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

        # 【重点修改】定义强大的数据增强流水线
        # 仅在训练模式(self.train=True)下启用
        if self.train:
            self.transform = A.Compose([
                # 1. 随机旋转：-45度到+45度
                A.Rotate(limit=45, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                # 2. 随机水平翻转
                A.HorizontalFlip(p=0.5),
                # 3. 随机垂直翻转
                A.VerticalFlip(p=0.5),
                # 4. 颜色抖动
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            ])
        else:
            self.transform = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".jpg"))
        png = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))
        
        #-------------------------------#
        #   数据增强 (Albumentations)
        #-------------------------------#
        if self.train and self.transform:
            # 将 PIL Image 转换为 numpy 数组
            jpg_np = np.array(jpg)
            png_np = np.array(png)

            # 【修复报错】强制对齐尺寸：如果原图和标签尺寸不一致，手动 Resize 标签
            if jpg_np.shape[0] != png_np.shape[0] or jpg_np.shape[1] != png_np.shape[1]:
                # cv2.resize 参数是 (width, height)
                png_np = cv2.resize(png_np, (jpg_np.shape[1], jpg_np.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 执行 Albumentations 变换
            transformed = self.transform(image=jpg_np, mask=png_np)
            jpg_np = transformed['image']
            png_np = transformed['mask']

            # 转回 PIL 以便后续兼容你原来的 get_random_data_basic 逻辑
            jpg = Image.fromarray(jpg_np)
            png = Image.fromarray(png_np)
            
        # 使用原来的函数进行最终的 Resize 和 Padding (Letterbox)，保证输入尺寸一致
        jpg, png = self.get_random_data_basic(jpg, png, self.input_shape, random=self.train)

        #-------------------------------#
        #   预处理与格式转换
        #-------------------------------#
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        png = np.array(png)
        # 这里的处理是将标签中超出类别的部分归类（通常是边缘线 255）
        png[png >= self.num_classes] = self.num_classes
        
        #-------------------------------------------------------#
        #   转化成 one_hot 的形式
        #-------------------------------------------------------#
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data_basic(self, image, label, input_shape, jitter=.3, random=True):
        image = cvtColor(image)
        # 确保 label 是 PIL 格式
        if not isinstance(label, Image.Image):
            label = Image.fromarray(np.array(label))
        
        iw, ih = image.size
        h, w   = input_shape

        # 验证集/测试集：保持纵横比的 Letterbox Resize
        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)

            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label = label.resize((nw,nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        # 训练集：带有随机长宽比抖动的 Resize
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.5, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
            
        nw = max(1, nw)
        nh = max(1, nh)

        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        
        # 随机位置填充（Random Cropping 效果）
        dx = int(self.rand(0, w-nw)) if w > nw else 0
        dy = int(self.rand(0, h-nh)) if h > nh else 0
        
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        
        return new_image, new_label

# DataLoader中collate_fn使用
def unet_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels