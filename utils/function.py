# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import logging
import os
import time
import cv2
from PIL import Image

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate



def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict, debug_summary_writer, train_dataset=None):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    if not train_dataset is None:
      train_dataset.get_transformed_image = True


    for i_iter, batch in enumerate(trainloader, 0):

        images, labels, bd_gts, _, _, transformed_images = batch
        transformed_labels = train_dataset.label2color(labels)

        # print(f"FUNCTION-LOG : Iteration : {i_iter}")
        # print(f"[FUNCTION-LOG] : image = {torch.unique(images)}")
        # print(f"[FUNCTION-LOG] : labels = {torch.unique(labels)}")
        # print(f"[FUNCTION-LOG] : bd_gts = {torch.unique(bd_gts)}")

        # print(f"---------------------------------------------")
        


        images = images.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()
        
        
        losses, _, acc, loss_list = model(images, labels, bd_gts, writer=debug_summary_writer, i_iter=i_iter, epoch=epoch, transformed_images=transformed_images,transformed_labels=transformed_labels, label2color=train_dataset.label2color)
        loss = losses.mean()
        acc  = acc.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        avg_sem_loss.update(loss_list[0].mean().item())
        avg_bce_loss.update(loss_list[1].mean().item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}, BCE loss: {:.6f}, SB loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                      ave_acc.average(), avg_sem_loss.average(), avg_bce_loss.average(),ave_loss.average()-avg_sem_loss.average()-avg_bce_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1
    if not train_dataset is None:
      train_dataset.get_transformed_image = False

def validate(config, testloader, model, writer_dict
            ,epoch,debug_summary_writer, test_dataset=None):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    test_dataset.get_transformed_image = True
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, bd_gts, _, _, transformed_images = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            bd_gts = bd_gts.float().cuda()
            losses, pred, _, _ = model(image, label, bd_gts,
                                      writer=debug_summary_writer,
                                      i_iter=idx,
                                      epoch=epoch,
                                      transformed_images=transformed_images,
                                      transformed_labels=None,
                                      label2color=test_dataset.label2color,
                                      output_tag="__VAL__"
                                      )
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            ave_loss.update(loss.item())

    test_dataset.get_transformed_image = False
    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        
        logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array


def testval(config, test_dataset, testloader, model,
            sv_dir='./', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, _, name = batch
            size = label.size()
            pred = test_dataset.single_scale_inference(config, model, image.cuda())

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def mask_overlay(image, mask):
  mask_rgb = np.array(Image.fromarray(mask))
  mask_rgba = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1], 4))
  mask_rgba[:,:,:3] = mask_rgb[:,:,[2,1,0]]
  mask_rgba[:,:,3] = 150

  if len(image.shape) == 3:
      image_rgba = np.zeros((image.shape[0],image.shape[1],4))
      image_rgba[:,:,3]=255


  # print(mask_rgba.shape)
  # print(mask_rgba[:10,:10,:10])
  # print(image.shape)

  result = cv2.addWeighted(image, 1, mask_rgba.astype(np.uint8), 0.3, 0)
  # result = mask_rgba
  return result


def test(config, test_dataset, testloader, model,
         sv_dir='./', sv_pred=True, img_dc="/content/shoga-sem-segmentation-14030515-4/test/"):
    model.eval()
    test_dataset.get_transformed_image=True
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, _, _, size, name, input_image = batch

            size = size[0]
            pred = test_dataset.single_scale_inference(
                config,
                model,
                image.cuda())

            
            # if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
            #     pred = F.interpolate(
            #         pred, size[-2:],
            #         mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            #     )
                
            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)

                # row,col = 1000,500
                # print(f"pred 0 : {pred[0,0,row:row+5,col:col+5]}")
                # print(f"pred 1 : {pred[0,1,row:row+5,col:col+5]}")
                # print(f"pred 2 : {pred[0,2,row:row+5,col:col+5]}")
                # print(f"pred 3 : {pred[0,3,row:row+5,col:col+5]}")
                # print(f"pred 4 : {pred[0,4,row:row+5,col:col+5]}")
                # print(f"pred 5 : {pred[0,5,row:row+5,col:col+5]}")

                pred = np.asarray(np.argmax(pred.cpu(), axis=1), dtype=np.uint8)
                pred = test_dataset.label2color(pred)


                load_img = cv2.cvtColor(cv2.imread(f"{img_dc}{name[0]}.png"), cv2.COLOR_RGB2RGBA)
                load_img[:,:,:3] = input_image[0].detach().numpy().astype(np.uint8)


                save_img = mask_overlay(load_img,
                                        pred[0,:,:,:]).astype(np.uint8)
                cv2.imwrite(os.path.join(sv_path, name[0]+'.jpg'), save_img)
    test_dataset.get_transformed_image=False
