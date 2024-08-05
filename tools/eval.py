# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
import pprint

import logging
import timeit


import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.function import testval, test
from utils.utils import create_logger
from PIL import Image

from torch.nn import functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--export',
                        help='Whether to export wrapped network as torch script',
                        default='none',
                        type=str,
                        )

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))


    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = models.pidnet.get_seg_model(config, imgnet_pretrained=True)

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'best.pt')      
   
    print(f"[EVAL] : model_state_file={model_state_file}")

    logger.info('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    
    start = timeit.default_timer()


    # model torch script export
    print(f"[EXPORT MODE] : {args.export}")
    if args.export == "torch-script":
      model   = PIDNetWrapper(core_address=config.TEST.MODEL_FILE)
      example, _, _, _, _ = next(iter(testloader))
      model.eval()
      traced_script_module = torch.jit.trace(model, example)

      output = traced_script_module(torch.ones(1, 3, 224, 224))

      img = Image.new('1', output.shape)
      pixels = img.load()
      for i in range(img.size[0]):
         for j in range(img.size[1]):
             pixels[i, j] = output[i][j]
      img.show()




      traced_script_module.save("/content/PIDNet/traced_pidnet_module.pt")
      return

    if ('test' in config.DATASET.TEST_SET) and ('city' in config.DATASET.DATASET):




    if True:
        print(f"[EVAL] : Final Output Directory : {final_output_dir}")
        test(config, 
             test_dataset, 
             testloader, 
             model,
             sv_dir=final_output_dir)
        
    else:
        mean_IoU, IoU_array, pixel_acc, mean_acc = testval(config, 
                                                           test_dataset, 
                                                           testloader, 
                                                           model)
    
        msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
            Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, 
            pixel_acc, mean_acc)
        logging.info(msg)
        logging.info(IoU_array)


    end = timeit.default_timer()
    logger.info('Mins: %d' % int((end-start)/60))
    logger.info('Done')

class PIDNetWrapper(nn.Module):
    def __init__(self, core_address):
        super(PIDNetWrapper, self).__init__()
        self.core = models.pidnet.get_seg_model(config, imgnet_pretrained=True)
        pretrained_dict = torch.load(core_address)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = self.core.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                            if k[6:] in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.core.load_state_dict(model_dict)
        self.color_list = [[0, 0, 0], [255, 255, 255],]

    def forward(self, x):
        pred = self.core(x)[1]

        print(f"[OUTPUT NUM] : {config.MODEL.NUM_OUTPUTS}")
        print(f"[OUTPUT INDEX] : {config.TEST.OUTPUT_INDEX}")
        if config.MODEL.NUM_OUTPUTS > 1:
              pred = pred[config.TEST.OUTPUT_INDEX]


        pred = pred.exp()
        pred = F.interpolate(
            input=pred, size=x.shape[-2:],
            mode='bilinear', align_corners=False
        )
        pred = np.asarray(np.argmax(pred.cpu(), axis=1), dtype=np.uint8).squeeze()

        color_map = np.zeros(pred.shape+(3,))
        for i, v in enumerate(self.color_list):
            color_map[pred==i] = self.color_list[i]

        print(f"[OUTPUT SHAPE] : Output shape is = {pred.shape}")

        return color_map



if __name__ == '__main__':
    main()

