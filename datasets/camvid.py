# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# Editted by KKM
# Adjust lines [47, 110, ] before using the script to train the models
# ------------------------------------------------------------------------------

import os
import numpy as np
from PIL import Image
import torch

from .base_dataset import BaseDataset

class CamVid(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_classes,
                 multi_scale, 
                 flip, 
                 ignore_label, 
                 base_size, 
                 crop_size,
                 scale_factor=1,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=0):

        super(CamVid, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()


        self.ignore_label = ignore_label
        

        # multi-class segmentation
        self.color_list = [
          [0, 0, 0],       # 0 : background
          [0, 255, 206],   # 1 : buttom
          [255, 255, 0],   # 2 : finish
          [0, 183, 235],   # 3 : neck
          [255, 0, 255],   # 4 : shoulder
          [255, 128, 0],   # 5 : side
        ]

        # # binary segmentation
        # self.color_list = [
        #   [0, 0, 0],       # 0 : background
        #   [255, 255, 255], # 1 : bottle
        # ]
        
        # classes must be counted inside the dataset
        self.class_weights =  None
        
        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self):
        files = []

        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })
            
        return files
        
    def color2label(self, color_map):
        label = np.ones(color_map.shape[:2])*self.ignore_label
        for i, v in enumerate(self.color_list):
            label[(color_map == v).sum(2)==3] = i

        return label.astype(np.uint8)
    
    def label2color(self, label):
        color_map = np.zeros(label.shape+(3,))
        for i, v in enumerate(self.color_list):
            color_map[label==i] = self.color_list[i]
            
        return color_map.astype(np.uint8)

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = Image.open(os.path.join(self.root,'camvid',item["img"])).convert('RGB')
        image = np.array(image)
        size = image.shape

        # print(f"[DL-LOG] : Image size is {image.shape}")

        color_map = Image.open(os.path.join(self.root,'camvid',item["label"])).convert('RGB')
        color_map = np.array(color_map)

        # print(f"[DL-LOG] : label item is {item['label']}")
        # print(f"[DL-LOG] : ColorMap unique values {np.unique(color_map)}")


        # # if mask data are not real class values use this to convert them
        # # comment out this line otherwise
        # label = self.color2label(color_map)
        label = color_map[0]
        

        # print(f"[DL-LOG] : ColorMap unique values {np.unique(label)}")

        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_pad=False,
                                edge_size=self.bd_dilate_size, city=False)

        # print(f"[DL-LOG] : image.shape = {image.shape}")
        # print(f"[DL-LOG] : label.shape = {label.shape}")
        # print(f"[DL-LOG] : edge.shape = {edge.shape}")

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
