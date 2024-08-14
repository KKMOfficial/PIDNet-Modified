# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path
from PIL import Image
import cv2

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config
import torchvision


def mask_overlay(image, mask):
  mask_rgb = np.array(Image.fromarray(mask))
  mask_rgba = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1], 4))
  mask_rgba[:,:,:3] = mask_rgb[:,:,[2,1,0]]
  mask_rgba[:,:,3] = 150

  if len(image.shape) == 3:
      image_rgba = np.array(Image.fromarray(image).convert("RGBA"))
  else: image_rgba = image


  result = cv2.addWeighted(image_rgba.astype(np.uint8), 1, mask_rgba.astype(np.uint8), 0.3, 0)
  
  return result



class FullModel(nn.Module):

  def __init__(self, model, sem_loss, bd_loss):
    super(FullModel, self).__init__()
    self.model = model
    self.sem_loss = sem_loss
    self.bd_loss = bd_loss

  def pixel_acc(self, pred, label):
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

  def forward(self, inputs, labels, bd_gt, writer=None, i_iter=None,epoch=None, transformed_images=None, transformed_labels=None, label2color=None, *args, **kwargs):
    
    outputs = self.model(inputs, *args, **kwargs)

    h, w = labels.size(1), labels.size(2)
    ph, pw = outputs[0].size(2), outputs[0].size(3)
    if ph != h or pw != w:
        for i in range(len(outputs)):
            outputs[i] = F.interpolate(outputs[i], size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

    # used semantic loss for two different purpose
    # log data here
    # print(f"[UTILS] : input shape is {inputs.shape}")
    # print(f"[UTILS] : transformed_images shape is {transformed_images.shape}")
    # print(f"[UTILS] : used labels is {(labels.shape[0],1,labels.shape[1],labels.shape[2])}")
    # print(f"[UTILS] : labels shape is {labels.shape}")
    # print(f"[UTILS] : transformed labels shape is {transformed_labels.shape}")
    # print(f"[UTILS] : outputs[-2] shape is {outputs[-2].shape}")
    # print(f"[UTILS] : outputs[-1] shape is {outputs[-1].shape}")
    # print(f"[UTILS] : output argmaxed = {np.argmax(outputs[-2].detach().cpu().numpy(), axis=1).shape}")
    # print(f"[UTILS] : output unique values = {np.unique(np.argmax(outputs[-2].detach().cpu().numpy(), axis=1))}")


    # print(f"[UTILS] : transformed output shape is {transformed_outputs.shape}")

    if (writer is not None) and (i_iter%20==1):
        grid_image = torchvision.utils.make_grid(transformed_images.permute((0,3,1,2)))[None,:,:,:]
        if not label2color is None:
            transformed_outputs = torch.tensor(label2color(np.argmax(outputs[-2].detach().cpu().numpy(), axis=1))).permute((0,3,1,2))
            # print(f"[UTILS] : transformed_outputs before gridify shape = {transformed_outputs.shape}")
            grid_transformed_output = torchvision.utils.make_grid(transformed_outputs)
            # print(f"[UTILS] : transformed_outputs after gridify shape = {grid_transformed_output.shape}")
            # print(f"[UTILS] : transformed_outputs after gridify shape = {grid_transformed_output.shape}")
            # print(f"[UTILS] : grid transformed output shape : {grid_transformed_output.detach().cpu().numpy().transpose((1,2,0)).astype(np.uint8).shape}")
            # print(f"[UTILS] : grid_image shape : {grid_image[0].detach().cpu().numpy().transpose((1,2,0)).astype(np.uint8).shape}")

            grid_overlay = mask_overlay(
              grid_image[0].detach().cpu().numpy().transpose((1,2,0)).astype(np.uint8),
              grid_transformed_output.detach().cpu().numpy().transpose((1,2,0)).astype(np.uint8)
            )
        if not transformed_images is None : writer.add_images(f"Pre-Infer/images-epoch{epoch}", grid_image, global_step=i_iter//10)
        if not transformed_labels is None : writer.add_images(f"Pre-Infer/images-epoch{epoch}", torchvision.utils.make_grid(torch.tensor(transformed_labels).permute((0,3,1,2)))[None,:,:,:], global_step=i_iter//10)
        writer.add_images(f"Post-Infer/Border-epoch{epoch}", torchvision.utils.make_grid(outputs[-1])[None,:,:,:], global_step=i_iter//10)
        writer.add_images(f"Post-Infer/Segment-epoch{epoch}", grid_transformed_output[None,:,:,:], global_step=i_iter//10)
        cv2.imwrite(f"/content/PIDNet/output/camvid/test/output_{i_iter}.jpg", grid_overlay)
        

    acc  = self.pixel_acc(outputs[-2], labels)
    loss_s = self.sem_loss(outputs[:-1], labels)

    loss_b = self.bd_loss(outputs[-1], bd_gt)

    filler = torch.ones_like(labels) * config.TRAIN.IGNORE_LABEL
    bd_label = torch.where(F.sigmoid(outputs[-1][:,0,:,:])>0.8, labels, filler)

    loss_sb = self.sem_loss(score=[outputs[-2]], target=bd_label)

    # print(f"loss_sb = {loss_sb}")

    loss = loss_s + loss_b + loss_sb

    return torch.unsqueeze(loss,0), outputs[:-1], acc, [loss_s, loss_b]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr
