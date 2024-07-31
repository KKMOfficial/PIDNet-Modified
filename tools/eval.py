
PIDNet.ipynb_
Notebook unstarred
Files
..
Drop files to upload them to session storage.
Dataset Download And Prepration

%%capture
!pip install roboflow
import zipfile

import os, requests
from roboflow import Roboflow
rf = Roboflow(api_key="yX1nJyVNr91CZv1vcBAf")
project = rf.workspace("minemy").project("shoga-segmentation-combined-14030118")
version = project.version(2)
dataset = version.download("coco")

# !pip install -q condacolab
# import condacolab
# condacolab.install()

!pip install wget
import os, requests, zipfile
import wget

!pip3 install torch torchvision torchaudio
!pip install opencv-python

/content
loading annotations into memory...
Done (t=0.02s)
creating index...
index created!

  2%|▏         | 4/238 [00:01<01:27,  2.68it/s]

[4 missed due to [image size mismatch].]

  4%|▍         | 9/238 [00:02<01:13,  3.11it/s]

[9 missed due to [image size mismatch].]
[10 missed due to [image size mismatch].]

  6%|▋         | 15/238 [00:04<01:08,  3.26it/s]

[15 missed due to [image size mismatch].]

  7%|▋         | 17/238 [00:04<00:56,  3.89it/s]

[17 missed due to [image size mismatch].]

  9%|▉         | 21/238 [00:05<01:01,  3.51it/s]

[21 missed due to [image size mismatch].]

 10%|▉         | 23/238 [00:06<00:52,  4.11it/s]

[23 missed due to [image size mismatch].]
[24 missed due to [image size mismatch].]

 11%|█▏        | 27/238 [00:07<00:45,  4.65it/s]

[27 missed due to [image size mismatch].]

 13%|█▎        | 30/238 [00:07<00:47,  4.34it/s]

[30 missed due to [image size mismatch].]

 14%|█▍        | 33/238 [00:08<00:51,  4.00it/s]

[33 missed due to [image size mismatch].]

 15%|█▍        | 35/238 [00:08<00:45,  4.50it/s]

[35 missed due to [image size mismatch].]
[36 missed due to [image size mismatch].]
[37 missed due to [image size mismatch].]

 17%|█▋        | 41/238 [00:09<00:43,  4.53it/s]

[41 missed due to [list index out of range].]

 19%|█▉        | 46/238 [00:11<01:07,  2.83it/s]

[46 missed due to [image size mismatch].]

 20%|██        | 48/238 [00:12<00:49,  3.81it/s]

[48 missed due to [image size mismatch].]

 21%|██▏       | 51/238 [00:12<00:47,  3.94it/s]

[51 missed due to [image size mismatch].]
[52 missed due to [image size mismatch].]
[53 missed due to [image size mismatch].]
[54 missed due to [image size mismatch].]
[55 missed due to [image size mismatch].]
[56 missed due to [image size mismatch].]

 24%|██▍       | 58/238 [00:13<00:20,  8.69it/s]

[58 missed due to [image size mismatch].]
[59 missed due to [image size mismatch].]

 26%|██▌       | 61/238 [00:13<00:20,  8.61it/s]

[61 missed due to [image size mismatch].]

 26%|██▋       | 63/238 [00:13<00:22,  7.79it/s]

[63 missed due to [image size mismatch].]
[64 missed due to [image size mismatch].]

 28%|██▊       | 66/238 [00:14<00:20,  8.49it/s]

[66 missed due to [image size mismatch].]
[67 missed due to [image size mismatch].]

 30%|███       | 72/238 [00:15<00:36,  4.52it/s]

[72 missed due to [image size mismatch].]
[73 missed due to [image size mismatch].]

 32%|███▏      | 77/238 [00:16<00:37,  4.30it/s]

[77 missed due to [image size mismatch].]

 33%|███▎      | 79/238 [00:16<00:33,  4.71it/s]

[79 missed due to [image size mismatch].]
[80 missed due to [image size mismatch].]

 34%|███▍      | 82/238 [00:17<00:26,  5.98it/s]

[82 missed due to [image size mismatch].]
[83 missed due to [image size mismatch].]

 36%|███▌      | 85/238 [00:17<00:23,  6.41it/s]

[85 missed due to [image size mismatch].]

 39%|███▉      | 93/238 [00:20<00:45,  3.17it/s]

[93 missed due to [image size mismatch].]
[94 missed due to [image size mismatch].]

 41%|████      | 97/238 [00:20<00:32,  4.32it/s]

[97 missed due to [image size mismatch].]

 42%|████▏     | 99/238 [00:21<00:28,  4.80it/s]

[99 missed due to [image size mismatch].]
[100 missed due to [image size mismatch].]

 43%|████▎     | 103/238 [00:22<00:30,  4.38it/s]

[103 missed due to [image size mismatch].]

 44%|████▍     | 105/238 [00:22<00:33,  4.03it/s]

[105 missed due to [image size mismatch].]

 47%|████▋     | 111/238 [00:24<00:40,  3.15it/s]

[111 missed due to [image size mismatch].]

 48%|████▊     | 114/238 [00:25<00:33,  3.67it/s]

[114 missed due to [image size mismatch].]
[115 missed due to [image size mismatch].]
[116 missed due to [image size mismatch].]

 52%|█████▏    | 124/238 [00:27<00:38,  2.97it/s]

[124 missed due to [image size mismatch].]

 53%|█████▎    | 127/238 [00:28<00:32,  3.37it/s]

[127 missed due to [image size mismatch].]

 55%|█████▌    | 131/238 [00:29<00:28,  3.69it/s]

[131 missed due to [image size mismatch].]

 56%|█████▌    | 133/238 [00:29<00:23,  4.38it/s]

[133 missed due to [image size mismatch].]

 57%|█████▋    | 135/238 [00:30<00:20,  4.98it/s]

[135 missed due to [image size mismatch].]
[136 missed due to [image size mismatch].]
[137 missed due to [image size mismatch].]
[138 missed due to [image size mismatch].]

 61%|██████    | 145/238 [00:32<00:25,  3.64it/s]

[145 missed due to [image size mismatch].]
[146 missed due to [image size mismatch].]
[147 missed due to [image size mismatch].]
[148 missed due to [image size mismatch].]

 63%|██████▎   | 150/238 [00:32<00:12,  6.80it/s]

[150 missed due to [image size mismatch].]
[151 missed due to [image size mismatch].]

 64%|██████▍   | 153/238 [00:33<00:11,  7.35it/s]

[153 missed due to [image size mismatch].]

 65%|██████▌   | 155/238 [00:33<00:11,  6.92it/s]

[155 missed due to [image size mismatch].]

 66%|██████▌   | 157/238 [00:33<00:12,  6.69it/s]

[157 missed due to [image size mismatch].]

 67%|██████▋   | 159/238 [00:34<00:12,  6.30it/s]

[159 missed due to [image size mismatch].]
[160 missed due to [image size mismatch].]
[161 missed due to [image size mismatch].]

 68%|██████▊   | 163/238 [00:34<00:11,  6.43it/s]

[163 missed due to [image size mismatch].]

 70%|██████▉   | 166/238 [00:35<00:17,  4.22it/s]

[166 missed due to [image size mismatch].]

 71%|███████   | 168/238 [00:36<00:16,  4.30it/s]

[168 missed due to [image size mismatch].]

 72%|███████▏  | 172/238 [00:37<00:17,  3.76it/s]

[172 missed due to [image size mismatch].]

 73%|███████▎  | 174/238 [00:37<00:14,  4.47it/s]

[174 missed due to [image size mismatch].]
[175 missed due to [image size mismatch].]

 74%|███████▍  | 177/238 [00:37<00:10,  5.84it/s]

[177 missed due to [image size mismatch].]
[178 missed due to [image size mismatch].]
[179 missed due to [image size mismatch].]
[180 missed due to [image size mismatch].]

 78%|███████▊  | 186/238 [00:39<00:11,  4.50it/s]

[186 missed due to [image size mismatch].]

 79%|███████▉  | 189/238 [00:40<00:10,  4.66it/s]

[189 missed due to [image size mismatch].]
[190 missed due to [image size mismatch].]

 81%|████████  | 193/238 [00:40<00:08,  5.27it/s]

[193 missed due to [image size mismatch].]

 82%|████████▏ | 195/238 [00:41<00:07,  5.55it/s]

[195 missed due to [image size mismatch].]

 87%|████████▋ | 206/238 [00:44<00:10,  2.94it/s]

[206 missed due to [image size mismatch].]
[207 missed due to [image size mismatch].]

 88%|████████▊ | 209/238 [00:45<00:06,  4.43it/s]

[209 missed due to [image size mismatch].]

 90%|████████▉ | 214/238 [00:46<00:08,  2.77it/s]

[214 missed due to [image size mismatch].]

 92%|█████████▏| 219/238 [00:48<00:07,  2.69it/s]

[219 missed due to [image size mismatch].]
[220 missed due to [image size mismatch].]

 94%|█████████▎| 223/238 [00:49<00:03,  4.12it/s]

[223 missed due to [image size mismatch].]

 96%|█████████▌| 228/238 [00:50<00:02,  3.39it/s]

[228 missed due to [image size mismatch].]

 97%|█████████▋| 230/238 [00:51<00:01,  4.10it/s]

[230 missed due to [image size mismatch].]
[231 missed due to [image size mismatch].]

 98%|█████████▊| 233/238 [00:51<00:00,  5.54it/s]

[233 missed due to [image size mismatch].]
[234 missed due to [image size mismatch].]
[235 missed due to [image size mismatch].]

100%|██████████| 238/238 [00:51<00:00,  4.60it/s]

[237 missed due to [image size mismatch].]


Model Preperatin And Training

%%capture
!git clone https://github.com/HRNet/HRNet-Semantic-Segmentation $SEG_ROOT
%cd HRNet-Semantic-Segmentation
!pip install -r requirements.txt
%cd ..
!rm -rf /content/HRNet-Semantic-Segmentation

%%capture
%cd /content/
!git clone https://github.com/KKMOfficial/PIDNet-Modified.git
os.rename("PIDNet-Modified", "PIDNet")
!rm -rf /content/PIDNet/data/camvid/images
!rm -rf /content/PIDNet/data/camvid/labels
!cp -r /content/shoga-segmentation-combined-14030118-2/train/images /content/PIDNet/data/camvid/images
!cp -r /content/shoga-segmentation-combined-14030118-2/train/segmentation/ /content/PIDNet/data/camvid/labels

import glob

directory_path = "/content/PIDNet/data/camvid/images"
img_file = glob.glob(directory_path + "/*")
directory_path = "/content/PIDNet/data/camvid/labels"
lbl_file = glob.glob(directory_path + "/*")

root = "/content/PIDNet/data/list/camvid/"
files = [
    f"{root}test.lst",
    f"{root}train.lst",
    f"{root}trainval.lst",
    f"{root}val.lst",
]
for addr in files:
  print(f"target file is {addr}")
  with open(addr,"w")as f:
    f.write("\n".join([img_file[i].replace("/content/PIDNet/data/camvid/", "")\
                       +" "+\
                       lbl_file[i].replace("/content/PIDNet/data/camvid/", "")for i in range(len(img_file))]))

target file is /content/PIDNet/data/list/camvid/test.lst
target file is /content/PIDNet/data/list/camvid/train.lst
target file is /content/PIDNet/data/list/camvid/trainval.lst
target file is /content/PIDNet/data/list/camvid/val.lst

import os, wget
check_point = "/content/PIDNet/output/camvid/pidnet_large_shoga/checkpoint.pth.tar"
configs = f"""CUDNN:
  BENCHMARK: true
  DETERMINISTIC: false
  ENABLED: true
GPUS: (0,)
OUTPUT_DIR: 'output'
LOG_DIR: 'log'
WORKERS: 1
PRINT_FREQ: 1

DATASET:
  DATASET: camvid
  ROOT: data/
  TEST_SET: 'list/camvid/val.lst'
  TRAIN_SET: 'list/camvid/trainval.lst'
  NUM_CLASSES: 2
MODEL:
  NAME: pidnet_m
  NUM_OUTPUTS: 2
  PRETRAINED: "pretrained_models/cityscapes/PIDNet_L_ImageNet.pth.tar"
LOSS:
  USE_OHEM: false
  OHEMTHRES: 0.9
  OHEMKEEP: 131072
  BALANCE_WEIGHTS: [0.4, 1.0]
  SB_WEIGHTS: 1.0
TRAIN:
  IMAGE_SIZE:
  - 2880
  - 1024
  BASE_SIZE: 1024
  BATCH_SIZE_PER_GPU: 4
  SHUFFLE: true
  BEGIN_EPOCH: 0
  END_EPOCH: 237
  RESUME: false
  OPTIMIZER: sgd
  LR: 0.005
  WD: 0.0005
  MOMENTUM: 0.9
  NESTEROV: false
  FLIP: true
  MULTI_SCALE: False
  IGNORE_LABEL: 255
  SCALE_FACTOR: 1
TEST:
  IMAGE_SIZE:
  - 2880
  - 1024
  BASE_SIZE: 1024
  BATCH_SIZE_PER_GPU: 1
  FLIP_TEST: false
  MULTI_SCALE: false
  MODEL_FILE: ''
  OUTPUT_INDEX: 1
"""
with open("/content/PIDNet/configs/camvid/pidnet_large_shoga.yaml","w") as f:
  f.write(configs)

def download_file(url, dest_dir, name):
  os.makedirs(dest_dir, exist_ok=True)
  wget.download(url,out = f"{dest_dir}{name}")


PIDNet_L_url = "https://drive.usercontent.google.com/download?id=1Eg6BwEsnu3AkKLO8lrKsoZ8AOEb2KZHY&export=download&authuser=0&confirm=t&uuid=8cd5038f-efcc-4b34-9cf0-0b2172270288&at=APZUnTWtRZY5mzGpgzuplxrEKxxy%3A1722240855017"
PIDNet_L_loc = "/content/PIDNet/pretrained_models/cityscapes/"
download_file(PIDNet_L_url, PIDNet_L_loc, "PIDNet_L_ImageNet.pth.tar")

Train The Models

# change number of the classes inside the camvid configuration files in order to process shoga dataset!
# /content/PIDNet/datasets/base_dataset.py : 80,84,87 : change np.int to int

%cd PIDNet/
!rm -rf /content/PIDNet/runs

[Errno 2] No such file or directory: 'PIDNet/'
/content/PIDNet

2024-07-31 11:12:45.377859: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-07-31 11:12:45.377913: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-07-31 11:12:45.379330: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-07-31 11:12:45.386519: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-07-31 11:12:46.590198: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Seeding with 304
=> creating output/camvid/pidnet_large_shoga
=> creating log/camvid/pidnet_m/pidnet_large_shoga_2024-07-31-11-12
Namespace(cfg='configs/camvid/pidnet_large_shoga.yaml', seed=304, opts=[])
AUTO_RESUME: False
CUDNN:
  BENCHMARK: True
  DETERMINISTIC: False
  ENABLED: True
DATASET:
  DATASET: camvid
  EXTRA_TRAIN_SET: 
  NUM_CLASSES: 2
  ROOT: data/
  TEST_SET: list/camvid/val.lst
  TRAIN_SET: list/camvid/trainval.lst
GPUS: (0,)
LOG_DIR: log
LOSS:
  BALANCE_WEIGHTS: [0.4, 1.0]
  CLASS_BALANCE: False
  OHEMKEEP: 131072
  OHEMTHRES: 0.9
  SB_WEIGHTS: 1.0
  USE_OHEM: False
MODEL:
  ALIGN_CORNERS: True
  NAME: pidnet_m
  NUM_OUTPUTS: 2
  PRETRAINED: pretrained_models/cityscapes/PIDNet_L_ImageNet.pth.tar
OUTPUT_DIR: output
PIN_MEMORY: True
PRINT_FREQ: 1
TEST:
  BASE_SIZE: 1024
  BATCH_SIZE_PER_GPU: 1
  FLIP_TEST: False
  IMAGE_SIZE: [2880, 1024]
  MODEL_FILE: 
  MULTI_SCALE: False
  OUTPUT_INDEX: 1
TRAIN:
  BASE_SIZE: 1024
  BATCH_SIZE_PER_GPU: 4
  BEGIN_EPOCH: 0
  END_EPOCH: 237
  EXTRA_EPOCH: 0
  EXTRA_LR: 0.001
  FLIP: True
  IGNORE_LABEL: 255
  IMAGE_SIZE: [2880, 1024]
  LR: 0.005
  MOMENTUM: 0.9
  MULTI_SCALE: False
  NESTEROV: False
  OPTIMIZER: sgd
  RESUME: False
  SCALE_FACTOR: 1
  SHUFFLE: True
  WD: 0.0005
WORKERS: 1
Attention!!!
Loaded 0 parameters!
Over!!!
/usr/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
Epoch: [0/237] Iter:[0/35], Time: 47.32, lr: [0.005], Loss: 7.698639, Acc:0.741919, Semantic loss: 4.571611, BCE loss: 0.611881, SB loss: 2.515146
Epoch: [0/237] Iter:[1/35], Time: 28.89, lr: [0.004999457501250636], Loss: 6.724725, Acc:0.791513, Semantic loss: 3.886269, BCE loss: 0.526147, SB loss: 2.312309
Epoch: [0/237] Iter:[2/35], Time: 22.61, lr: [0.004998914995960375], Loss: 6.555415, Acc:0.808672, Semantic loss: 3.716136, BCE loss: 0.436479, SB loss: 2.402801
Epoch: [0/237] Iter:[3/35], Time: 19.91, lr: [0.004998372484128349], Loss: 5.898695, Acc:0.822548, Semantic loss: 3.256024, BCE loss: 0.407908, SB loss: 2.234764
Epoch: [0/237] Iter:[4/35], Time: 18.21, lr: [0.00499782996575369], Loss: 6.100258, Acc:0.809318, Semantic loss: 3.255445, BCE loss: 0.375467, SB loss: 2.469346
Epoch: [0/237] Iter:[5/35], Time: 17.09, lr: [0.004997287440835531], Loss: 5.587748, Acc:0.820755, Semantic loss: 3.015920, BCE loss: 0.353751, SB loss: 2.218077
Epoch: [0/237] Iter:[6/35], Time: 16.08, lr: [0.004996744909373003], Loss: 5.246707, Acc:0.828140, Semantic loss: 2.822560, BCE loss: 0.337597, SB loss: 2.086550
Epoch: [0/237] Iter:[7/35], Time: 15.30, lr: [0.004996202371365236], Loss: 4.870012, Acc:0.834937, Semantic loss: 2.630102, BCE loss: 0.327245, SB loss: 1.912664
Epoch: [0/237] Iter:[8/35], Time: 14.86, lr: [0.0049956598268113645], Loss: 4.581885, Acc:0.842919, Semantic loss: 2.459421, BCE loss: 0.309810, SB loss: 1.812654
Epoch: [0/237] Iter:[9/35], Time: 14.46, lr: [0.004995117275710516], Loss: 4.308746, Acc:0.849171, Semantic loss: 2.301164, BCE loss: 0.302282, SB loss: 1.705300
Epoch: [0/237] Iter:[10/35], Time: 14.16, lr: [0.004994574718061826], Loss: 3.990453, Acc:0.858780, Semantic loss: 2.131182, BCE loss: 0.290195, SB loss: 1.569076
Epoch: [0/237] Iter:[11/35], Time: 13.75, lr: [0.00499403215386442], Loss: 3.731463, Acc:0.866898, Semantic loss: 1.986699, BCE loss: 0.282900, SB loss: 1.461864
Epoch: [0/237] Iter:[12/35], Time: 13.49, lr: [0.004993489583117431], Loss: 3.514622, Acc:0.872672, Semantic loss: 1.866996, BCE loss: 0.275639, SB loss: 1.371988
Epoch: [0/237] Iter:[13/35], Time: 13.31, lr: [0.00499294700581999], Loss: 3.314599, Acc:0.878781, Semantic loss: 1.758480, BCE loss: 0.266262, SB loss: 1.289857
Epoch: [0/237] Iter:[14/35], Time: 13.14, lr: [0.004992404421971225], Loss: 3.125145, Acc:0.886028, Semantic loss: 1.654029, BCE loss: 0.259674, SB loss: 1.211442
Epoch: [0/237] Iter:[15/35], Time: 12.92, lr: [0.004991861831570268], Loss: 2.971009, Acc:0.890378, Semantic loss: 1.571903, BCE loss: 0.253342, SB loss: 1.145764
Epoch: [0/237] Iter:[16/35], Time: 12.77, lr: [0.004991319234616246], Loss: 2.826958, Acc:0.895394, Semantic loss: 1.492716, BCE loss: 0.246374, SB loss: 1.087868
Epoch: [0/237] Iter:[17/35], Time: 12.69, lr: [0.0049907766311082895], Loss: 2.694810, Acc:0.900255, Semantic loss: 1.421085, BCE loss: 0.240500, SB loss: 1.033225
Epoch: [0/237] Iter:[18/35], Time: 12.60, lr: [0.004990234021045529], Loss: 2.581373, Acc:0.903779, Semantic loss: 1.361674, BCE loss: 0.232879, SB loss: 0.986820
Epoch: [0/237] Iter:[19/35], Time: 12.53, lr: [0.004989691404427091], Loss: 2.479010, Acc:0.907459, Semantic loss: 1.302887, BCE loss: 0.226238, SB loss: 0.949885
Epoch: [0/237] Iter:[20/35], Time: 12.40, lr: [0.004989148781252105], Loss: 2.382534, Acc:0.911217, Semantic loss: 1.250381, BCE loss: 0.219596, SB loss: 0.912557
Epoch: [0/237] Iter:[21/35], Time: 12.32, lr: [0.0049886061515197], Loss: 2.294791, Acc:0.914034, Semantic loss: 1.203946, BCE loss: 0.214037, SB loss: 0.876808
Epoch: [0/237] Iter:[22/35], Time: 12.26, lr: [0.004988063515229004], Loss: 2.214065, Acc:0.916695, Semantic loss: 1.160813, BCE loss: 0.208747, SB loss: 0.844506
Epoch: [0/237] Iter:[23/35], Time: 12.23, lr: [0.004987520872379144], Loss: 2.152284, Acc:0.917919, Semantic loss: 1.128107, BCE loss: 0.204578, SB loss: 0.819599
Epoch: [0/237] Iter:[24/35], Time: 12.15, lr: [0.004986978222969249], Loss: 2.096317, Acc:0.919013, Semantic loss: 1.098079, BCE loss: 0.200256, SB loss: 0.797982
Epoch: [0/237] Iter:[25/35], Time: 12.05, lr: [0.0049864355669984465], Loss: 2.028768, Acc:0.921552, Semantic loss: 1.060539, BCE loss: 0.195580, SB loss: 0.772649
Epoch: [0/237] Iter:[26/35], Time: 12.00, lr: [0.004985892904465864], Loss: 1.985886, Acc:0.919775, Semantic loss: 1.040594, BCE loss: 0.191338, SB loss: 0.753954
Epoch: [0/237] Iter:[27/35], Time: 11.97, lr: [0.004985350235370626], Loss: 1.931118, Acc:0.922061, Semantic loss: 1.009250, BCE loss: 0.186938, SB loss: 0.734930
Epoch: [0/237] Iter:[28/35], Time: 11.93, lr: [0.004984807559711864], Loss: 1.876263, Acc:0.924259, Semantic loss: 0.979218, BCE loss: 0.182770, SB loss: 0.714274
Epoch: [0/237] Iter:[29/35], Time: 11.84, lr: [0.0049842648774887], Loss: 1.827869, Acc:0.925544, Semantic loss: 0.952475, BCE loss: 0.178243, SB loss: 0.697152
Epoch: [0/237] Iter:[30/35], Time: 11.81, lr: [0.004983722188700263], Loss: 1.781109, Acc:0.927554, Semantic loss: 0.926574, BCE loss: 0.174301, SB loss: 0.680234
Epoch: [0/237] Iter:[31/35], Time: 11.78, lr: [0.004983179493345679], Loss: 1.742184, Acc:0.928744, Semantic loss: 0.904451, BCE loss: 0.170547, SB loss: 0.667186
Epoch: [0/237] Iter:[32/35], Time: 11.75, lr: [0.004982636791424074], Loss: 1.702863, Acc:0.930470, Semantic loss: 0.881939, BCE loss: 0.166812, SB loss: 0.654112
Epoch: [0/237] Iter:[33/35], Time: 11.65, lr: [0.004982094082934573], Loss: 1.662000, Acc:0.932127, Semantic loss: 0.860143, BCE loss: 0.163044, SB loss: 0.638813
Epoch: [0/237] Iter:[34/35], Time: 11.58, lr: [0.004981551367876302], Loss: 1.627414, Acc:0.933530, Semantic loss: 0.839561, BCE loss: 0.159465, SB loss: 0.628388
0
10
20
30
40
50
60
70
80
90
100
110
120
130
0 [0.83784981 0.35306582] 0.5954578142840187
1 [0.58042761 0.28254356] 0.4314855836395676
=> saving checkpoint to output/camvid/pidnet_large_shogacheckpoint.pth.tar
Loss: 4.592, MeanIU:  0.4315, Best_mIoU:  0.4315
[0.58042761 0.28254356]
Epoch: [1/237] Iter:[0/35], Time: 14.10, lr: [0.004981008646248387], Loss: 0.644617, Acc:0.942246, Semantic loss: 0.410618, BCE loss: 0.055437, SB loss: 0.178562
Epoch: [1/237] Iter:[1/35], Time: 12.74, lr: [0.004980465918049951], Loss: 0.505834, Acc:0.968081, Semantic loss: 0.272858, BCE loss: 0.050485, SB loss: 0.182491
Epoch: [1/237] Iter:[2/35], Time: 12.12, lr: [0.004979923183280121], Loss: 0.579990, Acc:0.968892, Semantic loss: 0.251810, BCE loss: 0.047345, SB loss: 0.280834
Epoch: [1/237] Iter:[3/35], Time: 11.34, lr: [0.004979380441938021], Loss: 0.524602, Acc:0.974331, Semantic loss: 0.218423, BCE loss: 0.046775, SB loss: 0.259403
Epoch: [1/237] Iter:[4/35], Time: 11.01, lr: [0.004978837694022776], Loss: 0.501324, Acc:0.977084, Semantic loss: 0.200414, BCE loss: 0.043869, SB loss: 0.257040
Epoch: [1/237] Iter:[5/35], Time: 11.03, lr: [0.0049782949395335094], Loss: 0.512759, Acc:0.978107, Semantic loss: 0.190519, BCE loss: 0.042604, SB loss: 0.279637
Epoch: [1/237] Iter:[6/35], Time: 10.99, lr: [0.004977752178469345], Loss: 0.526774, Acc:0.980211, Semantic loss: 0.179181, BCE loss: 0.041115, SB loss: 0.306477
Epoch: [1/237] Iter:[7/35], Time: 10.81, lr: [0.004977209410829408], Loss: 0.517125, Acc:0.981287, Semantic loss: 0.171825, BCE loss: 0.040064, SB loss: 0.305236
Epoch: [1/237] Iter:[8/35], Time: 10.69, lr: [0.00497666663661282], Loss: 0.535155, Acc:0.980028, Semantic loss: 0.176025, BCE loss: 0.038736, SB loss: 0.320394
Epoch: [1/237] Iter:[9/35], Time: 10.72, lr: [0.004976123855818706], Loss: 0.541723, Acc:0.980854, Semantic loss: 0.171557, BCE loss: 0.038138, SB loss: 0.332028
Epoch: [1/237] Iter:[10/35], Time: 10.72, lr: [0.004975581068446187], Loss: 0.524877, Acc:0.981924, Semantic loss: 0.166176, BCE loss: 0.038139, SB loss: 0.320562
Epoch: [1/237] Iter:[11/35], Time: 10.65, lr: [0.004975038274494388], Loss: 0.522273, Acc:0.982553, Semantic loss: 0.163669, BCE loss: 0.037692, SB loss: 0.320911
Epoch: [1/237] Iter:[12/35], Time: 10.55, lr: [0.004974495473962432], Loss: 0.516016, Acc:0.983309, Semantic loss: 0.158805, BCE loss: 0.037614, SB loss: 0.319597
Epoch: [1/237] Iter:[13/35], Time: 10.59, lr: [0.004973952666849439], Loss: 0.559867, Acc:0.979073, Semantic loss: 0.180566, BCE loss: 0.039200, SB loss: 0.340101
Epoch: [1/237] Iter:[14/35], Time: 10.58, lr: [0.004973409853154532], Loss: 0.546982, Acc:0.979872, Semantic loss: 0.175906, BCE loss: 0.038273, SB loss: 0.332803
Epoch: [1/237] Iter:[15/35], Time: 10.53, lr: [0.004972867032876835], Loss: 0.533323, Acc:0.980691, Semantic loss: 0.171556, BCE loss: 0.037501, SB loss: 0.324266
Epoch: [1/237] Iter:[16/35], Time: 10.49, lr: [0.004972324206015468], Loss: 0.526182, Acc:0.981144, Semantic loss: 0.170177, BCE loss: 0.036883, SB loss: 0.319122
Epoch: [1/237] Iter:[17/35], Time: 10.53, lr: [0.004971781372569552], Loss: 0.524834, Acc:0.981691, Semantic loss: 0.168289, BCE loss: 0.036701, SB loss: 0.319844
Epoch: [1/237] Iter:[18/35], Time: 10.54, lr: [0.004971238532538209], Loss: 0.542493, Acc:0.981123, Semantic loss: 0.169454, BCE loss: 0.036636, SB loss: 0.336403
Epoch: [1/237] Iter:[19/35], Time: 10.52, lr: [0.004970695685920561], Loss: 0.550990, Acc:0.981090, Semantic loss: 0.169103, BCE loss: 0.036476, SB loss: 0.345411
Epoch: [1/237] Iter:[20/35], Time: 10.47, lr: [0.004970152832715727], Loss: 0.544196, Acc:0.981186, Semantic loss: 0.167884, BCE loss: 0.036109, SB loss: 0.340204
Epoch: [1/237] Iter:[21/35], Time: 10.49, lr: [0.00496960997292283], Loss: 0.536065, Acc:0.981565, Semantic loss: 0.165946, BCE loss: 0.035363, SB loss: 0.334757
Epoch: [1/237] Iter:[22/35], Time: 10.50, lr: [0.0049690671065409885], Loss: 0.539926, Acc:0.981258, Semantic loss: 0.167355, BCE loss: 0.034920, SB loss: 0.337651
Epoch: [1/237] Iter:[23/35], Time: 10.48, lr: [0.004968524233569323], Loss: 0.535576, Acc:0.981323, Semantic loss: 0.166750, BCE loss: 0.034271, SB loss: 0.334555
Epoch: [1/237] Iter:[24/35], Time: 10.45, lr: [0.004967981354006954], Loss: 0.527826, Acc:0.981662, Semantic loss: 0.165647, BCE loss: 0.033870, SB loss: 0.328309
Epoch: [1/237] Iter:[25/35], Time: 10.48, lr: [0.0049674384678530005], Loss: 0.526928, Acc:0.982122, Semantic loss: 0.163724, BCE loss: 0.033629, SB loss: 0.329575
Epoch: [1/237] Iter:[26/35], Time: 10.49, lr: [0.004966895575106583], Loss: 0.526008, Acc:0.982121, Semantic loss: 0.164117, BCE loss: 0.033390, SB loss: 0.328501
Epoch: [1/237] Iter:[27/35], Time: 10.50, lr: [0.0049663526757668195], Loss: 0.519483, Acc:0.982564, Semantic loss: 0.162512, BCE loss: 0.033202, SB loss: 0.323769
Epoch: [1/237] Iter:[28/35], Time: 10.46, lr: [0.004965809769832831], Loss: 0.518915, Acc:0.982112, Semantic loss: 0.164644, BCE loss: 0.032857, SB loss: 0.321414
Epoch: [1/237] Iter:[29/35], Time: 10.48, lr: [0.004965266857303734], Loss: 0.514863, Acc:0.982517, Semantic loss: 0.162559, BCE loss: 0.032655, SB loss: 0.319648
Epoch: [1/237] Iter:[30/35], Time: 10.48, lr: [0.00496472393817865], Loss: 0.514656, Acc:0.982920, Semantic loss: 0.159958, BCE loss: 0.032508, SB loss: 0.322189
Epoch: [1/237] Iter:[31/35], Time: 10.48, lr: [0.004964181012456695], Loss: 0.509137, Acc:0.983202, Semantic loss: 0.157733, BCE loss: 0.032167, SB loss: 0.319238
Epoch: [1/237] Iter:[32/35], Time: 10.44, lr: [0.004963638080136988], Loss: 0.502473, Acc:0.983601, Semantic loss: 0.156120, BCE loss: 0.031825, SB loss: 0.314528
Epoch: [1/237] Iter:[33/35], Time: 10.42, lr: [0.004963095141218646], Loss: 0.502889, Acc:0.983618, Semantic loss: 0.156390, BCE loss: 0.031579, SB loss: 0.314920
Epoch: [1/237] Iter:[34/35], Time: 10.41, lr: [0.004962552195700788], Loss: 0.500628, Acc:0.983852, Semantic loss: 0.154865, BCE loss: 0.031343, SB loss: 0.314420
=> saving checkpoint to output/camvid/pidnet_large_shogacheckpoint.pth.tar
Loss: 4.592, MeanIU:  0.4315, Best_mIoU:  0.4315
[0.58042761 0.28254356]
Epoch: [2/237] Iter:[0/35], Time: 13.18, lr: [0.004962009243582531], Loss: 0.400572, Acc:0.971145, Semantic loss: 0.165406, BCE loss: 0.015116, SB loss: 0.220050
Epoch: [2/237] Iter:[1/35], Time: 11.83, lr: [0.004961466284862992], Loss: 0.370193, Acc:0.982286, Semantic loss: 0.138464, BCE loss: 0.018506, SB loss: 0.213222
Epoch: [2/237] Iter:[2/35], Time: 11.03, lr: [0.004960923319541289], Loss: 0.354650, Acc:0.986690, Semantic loss: 0.121973, BCE loss: 0.020424, SB loss: 0.212253
Epoch: [2/237] Iter:[3/35], Time: 10.99, lr: [0.0049603803476165375], Loss: 0.370552, Acc:0.988659, Semantic loss: 0.122551, BCE loss: 0.021556, SB loss: 0.226445
Epoch: [2/237] Iter:[4/35], Time: 10.95, lr: [0.004959837369087856], Loss: 0.364571, Acc:0.989077, Semantic loss: 0.123772, BCE loss: 0.021618, SB loss: 0.219181
Epoch: [2/237] Iter:[5/35], Time: 10.90, lr: [0.004959294383954358], Loss: 0.349983, Acc:0.989870, Semantic loss: 0.117775, BCE loss: 0.020719, SB loss: 0.211489
Epoch: [2/237] Iter:[6/35], Time: 10.68, lr: [0.004958751392215162], Loss: 0.340338, Acc:0.990890, Semantic loss: 0.114551, BCE loss: 0.020193, SB loss: 0.205595
Epoch: [2/237] Iter:[7/35], Time: 10.65, lr: [0.004958208393869384], Loss: 0.333315, Acc:0.991656, Semantic loss: 0.110464, BCE loss: 0.019374, SB loss: 0.203477
Epoch: [2/237] Iter:[8/35], Time: 10.65, lr: [0.004957665388916137], Loss: 0.333640, Acc:0.992198, Semantic loss: 0.106754, BCE loss: 0.019069, SB loss: 0.207817
Epoch: [2/237] Iter:[9/35], Time: 10.68, lr: [0.0049571223773545395], Loss: 0.335424, Acc:0.991569, Semantic loss: 0.109796, BCE loss: 0.019406, SB loss: 0.206222
Epoch: [2/237] Iter:[10/35], Time: 10.55, lr: [0.004956579359183705], Loss: 0.331044, Acc:0.991900, Semantic loss: 0.107274, BCE loss: 0.019578, SB loss: 0.204192
Epoch: [2/237] Iter:[11/35], Time: 10.56, lr: [0.004956036334402749], Loss: 0.346526, Acc:0.991919, Semantic loss: 0.107736, BCE loss: 0.019799, SB loss: 0.218992
Epoch: [2/237] Iter:[12/35], Time: 10.59, lr: [0.0049554933030107875], Loss: 0.342472, Acc:0.992371, Semantic loss: 0.106583, BCE loss: 0.019629, SB loss: 0.216260
Epoch: [2/237] Iter:[13/35], Time: 10.60, lr: [0.004954950265006933], Loss: 0.339594, Acc:0.992741, Semantic loss: 0.103392, BCE loss: 0.019096, SB loss: 0.217106
Epoch: [2/237] Iter:[14/35], Time: 10.53, lr: [0.0049544072203903], Loss: 0.341018, Acc:0.992458, Semantic loss: 0.105819, BCE loss: 0.019003, SB loss: 0.216196
Epoch: [2/237] Iter:[15/35], Time: 10.50, lr: [0.004953864169160004], Loss: 0.338419, Acc:0.992759, Semantic loss: 0.102536, BCE loss: 0.018658, SB loss: 0.217225
Epoch: [2/237] Iter:[16/35], Time: 10.52, lr: [0.004953321111315159], Loss: 0.336624, Acc:0.992834, Semantic loss: 0.101764, BCE loss: 0.018782, SB loss: 0.216079
Epoch: [2/237] Iter:[17/35], Time: 10.54, lr: [0.0049527780468548764], Loss: 0.342300, Acc:0.991692, Semantic loss: 0.107607, BCE loss: 0.018746, SB loss: 0.215947
Epoch: [2/237] Iter:[18/35], Time: 10.49, lr: [0.004952234975778272], Loss: 0.339672, Acc:0.991968, Semantic loss: 0.106457, BCE loss: 0.018655, SB loss: 0.214560
Epoch: [2/237] Iter:[19/35], Time: 10.49, lr: [0.004951691898084458], Loss: 0.340233, Acc:0.992168, Semantic loss: 0.105769, BCE loss: 0.018563, SB loss: 0.215901
Epoch: [2/237] Iter:[20/35], Time: 10.52, lr: [0.004951148813772547], Loss: 0.338602, Acc:0.992373, Semantic loss: 0.104578, BCE loss: 0.018591, SB loss: 0.215434
Epoch: [2/237] Iter:[21/35], Time: 10.53, lr: [0.004950605722841652], Loss: 0.336473, Acc:0.992596, Semantic loss: 0.104089, BCE loss: 0.018500, SB loss: 0.213884
Epoch: [2/237] Iter:[22/35], Time: 10.50, lr: [0.004950062625290887], Loss: 0.339660, Acc:0.992638, Semantic loss: 0.103000, BCE loss: 0.018476, SB loss: 0.218183
Epoch: [2/237] Iter:[23/35], Time: 10.48, lr: [0.0049495195211193615], Loss: 0.338266, Acc:0.992780, Semantic loss: 0.103015, BCE loss: 0.018863, SB loss: 0.216388
Epoch: [2/237] Iter:[24/35], Time: 10.51, lr: [0.00494897641032619], Loss: 0.343450, Acc:0.992010, Semantic loss: 0.106523, BCE loss: 0.018792, SB loss: 0.218135
Epoch: [2/237] Iter:[25/35], Time: 10.51, lr: [0.004948433292910483], Loss: 0.338792, Acc:0.991787, Semantic loss: 0.105470, BCE loss: 0.018559, SB loss: 0.214762
Epoch: [2/237] Iter:[26/35], Time: 10.48, lr: [0.004947890168871353], Loss: 0.341856, Acc:0.991711, Semantic loss: 0.108071, BCE loss: 0.018638, SB loss: 0.215147
Epoch: [2/237] Iter:[27/35], Time: 10.45, lr: [0.0049473470382079094], Loss: 0.340324, Acc:0.991896, Semantic loss: 0.107618, BCE loss: 0.018634, SB loss: 0.214072
Epoch: [2/237] Iter:[28/35], Time: 10.47, lr: [0.004946803900919266], Loss: 0.337353, Acc:0.991715, Semantic loss: 0.107352, BCE loss: 0.018500, SB loss: 0.211501
Epoch: [2/237] Iter:[29/35], Time: 10.48, lr: [0.004946260757004533], Loss: 0.336491, Acc:0.991889, Semantic loss: 0.107324, BCE loss: 0.018499, SB loss: 0.210668
Epoch: [2/237] Iter:[30/35], Time: 10.44, lr: [0.00494571760646282], Loss: 0.335434, Acc:0.992040, Semantic loss: 0.107198, BCE loss: 0.018380, SB loss: 0.209855
Epoch: [2/237] Iter:[31/35], Time: 10.43, lr: [0.004945174449293237], Loss: 0.333544, Acc:0.992178, Semantic loss: 0.106901, BCE loss: 0.018346, SB loss: 0.208297
Epoch: [2/237] Iter:[32/35], Time: 10.44, lr: [0.0049446312854948965], Loss: 0.333675, Acc:0.992271, Semantic loss: 0.107391, BCE loss: 0.018375, SB loss: 0.207910
Epoch: [2/237] Iter:[33/35], Time: 10.41, lr: [0.0049440881150669075], Loss: 0.333484, Acc:0.992417, Semantic loss: 0.107317, BCE loss: 0.018328, SB loss: 0.207840
Epoch: [2/237] Iter:[34/35], Time: 10.35, lr: [0.0049435449380083795], Loss: 0.331396, Acc:0.992569, Semantic loss: 0.107001, BCE loss: 0.018308, SB loss: 0.206087
=> saving checkpoint to output/camvid/pidnet_large_shogacheckpoint.pth.tar
Loss: 4.592, MeanIU:  0.4315, Best_mIoU:  0.4315
[0.58042761 0.28254356]
Epoch: [3/237] Iter:[0/35], Time: 12.98, lr: [0.004943001754318422], Loss: 0.391982, Acc:0.995595, Semantic loss: 0.101964, BCE loss: 0.013163, SB loss: 0.276855
Epoch: [3/237] Iter:[1/35], Time: 11.96, lr: [0.004942458563996144], Loss: 0.335525, Acc:0.994166, Semantic loss: 0.103998, BCE loss: 0.013221, SB loss: 0.218306
Epoch: [3/237] Iter:[2/35], Time: 11.40, lr: [0.0049419153670406555], Loss: 0.303822, Acc:0.995088, Semantic loss: 0.088240, BCE loss: 0.013839, SB loss: 0.201743
Traceback (most recent call last):
  File "/content/PIDNet/tools/train.py", line 228, in <module>
    main()
  File "/content/PIDNet/tools/train.py", line 190, in main
    train(config, epoch, config.TRAIN.END_EPOCH, 
  File "/content/PIDNet/tools/../utils/function.py", line 55, in train
    losses, _, acc, loss_list = model(images, labels, bd_gts, writer=debug_summary_writer, i_iter=i_iter, epoch=epoch)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/parallel/data_parallel.py", line 183, in forward
    return self.module(*inputs[0], **module_kwargs[0])
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/content/PIDNet/tools/../utils/utils.py", line 54, in forward
    writer.add_images(f"Pre-Enter/images-epoch{epoch}", torchvision.utils.make_grid(inputs)[None,:,:,:], global_step=i_iter)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/tensorboard/writer.py", line 704, in add_images
    image(tag, img_tensor, dataformats=dataformats), global_step, walltime
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/tensorboard/summary.py", line 569, in image
    tensor = make_np(tensor)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/tensorboard/_convert_np.py", line 23, in make_np
    return _prepare_pytorch(x)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/tensorboard/_convert_np.py", line 32, in _prepare_pytorch
    x = x.detach().cpu().numpy()
KeyboardInterrupt

%reload_ext tensorboard
%tensorboard --logdir "/content/PIDNet/runs/Jul31_11-12-47_9d835d73a82b"  --samples_per_plugin "images=10000"

=> creating output/camvid/pidnet_large_shoga
=> creating log/camvid/pidnet_m/pidnet_large_shoga_2024-07-31-11-44
Namespace(cfg='configs/camvid/pidnet_large_shoga.yaml', opts=['TEST.MODEL_FILE', '/content/PIDNet/output/camvid/pidnet_large_shoga/best.pt', 'DATASET.TEST_SET', 'list/camvid/test.lst'])
CfgNode({'OUTPUT_DIR': 'output', 'LOG_DIR': 'log', 'GPUS': (0,), 'WORKERS': 1, 'PRINT_FREQ': 1, 'AUTO_RESUME': False, 'PIN_MEMORY': True, 'CUDNN': CfgNode({'BENCHMARK': True, 'DETERMINISTIC': False, 'ENABLED': True}), 'MODEL': CfgNode({'NAME': 'pidnet_m', 'PRETRAINED': 'pretrained_models/cityscapes/PIDNet_L_ImageNet.pth.tar', 'ALIGN_CORNERS': True, 'NUM_OUTPUTS': 2}), 'LOSS': CfgNode({'USE_OHEM': False, 'OHEMTHRES': 0.9, 'OHEMKEEP': 131072, 'CLASS_BALANCE': False, 'BALANCE_WEIGHTS': [0.4, 1.0], 'SB_WEIGHTS': 1.0}), 'DATASET': CfgNode({'ROOT': 'data/', 'DATASET': 'camvid', 'NUM_CLASSES': 2, 'TRAIN_SET': 'list/camvid/trainval.lst', 'EXTRA_TRAIN_SET': '', 'TEST_SET': 'list/camvid/test.lst'}), 'TRAIN': CfgNode({'IMAGE_SIZE': [2880, 1024], 'BASE_SIZE': 1024, 'FLIP': True, 'MULTI_SCALE': False, 'SCALE_FACTOR': 1, 'LR': 0.005, 'EXTRA_LR': 0.001, 'OPTIMIZER': 'sgd', 'MOMENTUM': 0.9, 'WD': 0.0005, 'NESTEROV': False, 'IGNORE_LABEL': 255, 'BEGIN_EPOCH': 0, 'END_EPOCH': 237, 'EXTRA_EPOCH': 0, 'RESUME': False, 'BATCH_SIZE_PER_GPU': 4, 'SHUFFLE': True}), 'TEST': CfgNode({'IMAGE_SIZE': [2880, 1024], 'BASE_SIZE': 1024, 'BATCH_SIZE_PER_GPU': 1, 'MODEL_FILE': '/content/PIDNet/output/camvid/pidnet_large_shoga/best.pt', 'FLIP_TEST': False, 'MULTI_SCALE': False, 'OUTPUT_INDEX': 1})})
Attention!!!
Loaded 335 parameters!
Over!!!
=> loading model from /content/PIDNet/output/camvid/pidnet_large_shoga/best.pt
=> loading conv1.0.weight from pretrained model
=> loading conv1.0.bias from pretrained model
=> loading conv1.1.weight from pretrained model
=> loading conv1.1.bias from pretrained model
=> loading conv1.1.running_mean from pretrained model
=> loading conv1.1.running_var from pretrained model
=> loading conv1.1.num_batches_tracked from pretrained model
=> loading conv1.3.weight from pretrained model
=> loading conv1.3.bias from pretrained model
=> loading conv1.4.weight from pretrained model
=> loading conv1.4.bias from pretrained model
=> loading conv1.4.running_mean from pretrained model
=> loading conv1.4.running_var from pretrained model
=> loading conv1.4.num_batches_tracked from pretrained model
=> loading layer1.0.conv1.weight from pretrained model
=> loading layer1.0.bn1.weight from pretrained model
=> loading layer1.0.bn1.bias from pretrained model
=> loading layer1.0.bn1.running_mean from pretrained model
=> loading layer1.0.bn1.running_var from pretrained model
=> loading layer1.0.bn1.num_batches_tracked from pretrained model
=> loading layer1.0.conv2.weight from pretrained model
=> loading layer1.0.bn2.weight from pretrained model
=> loading layer1.0.bn2.bias from pretrained model
=> loading layer1.0.bn2.running_mean from pretrained model
=> loading layer1.0.bn2.running_var from pretrained model
=> loading layer1.0.bn2.num_batches_tracked from pretrained model
=> loading layer1.1.conv1.weight from pretrained model
=> loading layer1.1.bn1.weight from pretrained model
=> loading layer1.1.bn1.bias from pretrained model
=> loading layer1.1.bn1.running_mean from pretrained model
=> loading layer1.1.bn1.running_var from pretrained model
=> loading layer1.1.bn1.num_batches_tracked from pretrained model
=> loading layer1.1.conv2.weight from pretrained model
=> loading layer1.1.bn2.weight from pretrained model
=> loading layer1.1.bn2.bias from pretrained model
=> loading layer1.1.bn2.running_mean from pretrained model
=> loading layer1.1.bn2.running_var from pretrained model
=> loading layer1.1.bn2.num_batches_tracked from pretrained model
=> loading layer2.0.conv1.weight from pretrained model
=> loading layer2.0.bn1.weight from pretrained model
=> loading layer2.0.bn1.bias from pretrained model
=> loading layer2.0.bn1.running_mean from pretrained model
=> loading layer2.0.bn1.running_var from pretrained model
=> loading layer2.0.bn1.num_batches_tracked from pretrained model
=> loading layer2.0.conv2.weight from pretrained model
=> loading layer2.0.bn2.weight from pretrained model
=> loading layer2.0.bn2.bias from pretrained model
=> loading layer2.0.bn2.running_mean from pretrained model
=> loading layer2.0.bn2.running_var from pretrained model
=> loading layer2.0.bn2.num_batches_tracked from pretrained model
=> loading layer2.0.downsample.0.weight from pretrained model
=> loading layer2.0.downsample.1.weight from pretrained model
=> loading layer2.0.downsample.1.bias from pretrained model
=> loading layer2.0.downsample.1.running_mean from pretrained model
=> loading layer2.0.downsample.1.running_var from pretrained model
=> loading layer2.0.downsample.1.num_batches_tracked from pretrained model
=> loading layer2.1.conv1.weight from pretrained model
=> loading layer2.1.bn1.weight from pretrained model
=> loading layer2.1.bn1.bias from pretrained model
=> loading layer2.1.bn1.running_mean from pretrained model
=> loading layer2.1.bn1.running_var from pretrained model
=> loading layer2.1.bn1.num_batches_tracked from pretrained model
=> loading layer2.1.conv2.weight from pretrained model
=> loading layer2.1.bn2.weight from pretrained model
=> loading layer2.1.bn2.bias from pretrained model
=> loading layer2.1.bn2.running_mean from pretrained model
=> loading layer2.1.bn2.running_var from pretrained model
=> loading layer2.1.bn2.num_batches_tracked from pretrained model
=> loading layer3.0.conv1.weight from pretrained model
=> loading layer3.0.bn1.weight from pretrained model
=> loading layer3.0.bn1.bias from pretrained model
=> loading layer3.0.bn1.running_mean from pretrained model
=> loading layer3.0.bn1.running_var from pretrained model
=> loading layer3.0.bn1.num_batches_tracked from pretrained model
=> loading layer3.0.conv2.weight from pretrained model
=> loading layer3.0.bn2.weight from pretrained model
=> loading layer3.0.bn2.bias from pretrained model
=> loading layer3.0.bn2.running_mean from pretrained model
=> loading layer3.0.bn2.running_var from pretrained model
=> loading layer3.0.bn2.num_batches_tracked from pretrained model
=> loading layer3.0.downsample.0.weight from pretrained model
=> loading layer3.0.downsample.1.weight from pretrained model
=> loading layer3.0.downsample.1.bias from pretrained model
=> loading layer3.0.downsample.1.running_mean from pretrained model
=> loading layer3.0.downsample.1.running_var from pretrained model
=> loading layer3.0.downsample.1.num_batches_tracked from pretrained model
=> loading layer3.1.conv1.weight from pretrained model
=> loading layer3.1.bn1.weight from pretrained model
=> loading layer3.1.bn1.bias from pretrained model
=> loading layer3.1.bn1.running_mean from pretrained model
=> loading layer3.1.bn1.running_var from pretrained model
=> loading layer3.1.bn1.num_batches_tracked from pretrained model
=> loading layer3.1.conv2.weight from pretrained model
=> loading layer3.1.bn2.weight from pretrained model
=> loading layer3.1.bn2.bias from pretrained model
=> loading layer3.1.bn2.running_mean from pretrained model
=> loading layer3.1.bn2.running_var from pretrained model
=> loading layer3.1.bn2.num_batches_tracked from pretrained model
=> loading layer3.2.conv1.weight from pretrained model
=> loading layer3.2.bn1.weight from pretrained model
=> loading layer3.2.bn1.bias from pretrained model
=> loading layer3.2.bn1.running_mean from pretrained model
=> loading layer3.2.bn1.running_var from pretrained model
=> loading layer3.2.bn1.num_batches_tracked from pretrained model
=> loading layer3.2.conv2.weight from pretrained model
=> loading layer3.2.bn2.weight from pretrained model
=> loading layer3.2.bn2.bias from pretrained model
=> loading layer3.2.bn2.running_mean from pretrained model
=> loading layer3.2.bn2.running_var from pretrained model
=> loading layer3.2.bn2.num_batches_tracked from pretrained model
=> loading layer4.0.conv1.weight from pretrained model
=> loading layer4.0.bn1.weight from pretrained model
=> loading layer4.0.bn1.bias from pretrained model
=> loading layer4.0.bn1.running_mean from pretrained model
=> loading layer4.0.bn1.running_var from pretrained model
=> loading layer4.0.bn1.num_batches_tracked from pretrained model
=> loading layer4.0.conv2.weight from pretrained model
=> loading layer4.0.bn2.weight from pretrained model
=> loading layer4.0.bn2.bias from pretrained model
=> loading layer4.0.bn2.running_mean from pretrained model
=> loading layer4.0.bn2.running_var from pretrained model
=> loading layer4.0.bn2.num_batches_tracked from pretrained model
=> loading layer4.0.downsample.0.weight from pretrained model
=> loading layer4.0.downsample.1.weight from pretrained model
=> loading layer4.0.downsample.1.bias from pretrained model
=> loading layer4.0.downsample.1.running_mean from pretrained model
=> loading layer4.0.downsample.1.running_var from pretrained model
=> loading layer4.0.downsample.1.num_batches_tracked from pretrained model
=> loading layer4.1.conv1.weight from pretrained model
=> loading layer4.1.bn1.weight from pretrained model
=> loading layer4.1.bn1.bias from pretrained model
=> loading layer4.1.bn1.running_mean from pretrained model
=> loading layer4.1.bn1.running_var from pretrained model
=> loading layer4.1.bn1.num_batches_tracked from pretrained model
=> loading layer4.1.conv2.weight from pretrained model
=> loading layer4.1.bn2.weight from pretrained model
=> loading layer4.1.bn2.bias from pretrained model
=> loading layer4.1.bn2.running_mean from pretrained model
=> loading layer4.1.bn2.running_var from pretrained model
=> loading layer4.1.bn2.num_batches_tracked from pretrained model
=> loading layer4.2.conv1.weight from pretrained model
=> loading layer4.2.bn1.weight from pretrained model
=> loading layer4.2.bn1.bias from pretrained model
=> loading layer4.2.bn1.running_mean from pretrained model
=> loading layer4.2.bn1.running_var from pretrained model
=> loading layer4.2.bn1.num_batches_tracked from pretrained model
=> loading layer4.2.conv2.weight from pretrained model
=> loading layer4.2.bn2.weight from pretrained model
=> loading layer4.2.bn2.bias from pretrained model
=> loading layer4.2.bn2.running_mean from pretrained model
=> loading layer4.2.bn2.running_var from pretrained model
=> loading layer4.2.bn2.num_batches_tracked from pretrained model
=> loading layer5.0.conv1.weight from pretrained model
=> loading layer5.0.bn1.weight from pretrained model
=> loading layer5.0.bn1.bias from pretrained model
=> loading layer5.0.bn1.running_mean from pretrained model
=> loading layer5.0.bn1.running_var from pretrained model
=> loading layer5.0.bn1.num_batches_tracked from pretrained model
=> loading layer5.0.conv2.weight from pretrained model
=> loading layer5.0.bn2.weight from pretrained model
=> loading layer5.0.bn2.bias from pretrained model
=> loading layer5.0.bn2.running_mean from pretrained model
=> loading layer5.0.bn2.running_var from pretrained model
=> loading layer5.0.bn2.num_batches_tracked from pretrained model
=> loading layer5.0.conv3.weight from pretrained model
=> loading layer5.0.bn3.weight from pretrained model
=> loading layer5.0.bn3.bias from pretrained model
=> loading layer5.0.bn3.running_mean from pretrained model
=> loading layer5.0.bn3.running_var from pretrained model
=> loading layer5.0.bn3.num_batches_tracked from pretrained model
=> loading layer5.0.downsample.0.weight from pretrained model
=> loading layer5.0.downsample.1.weight from pretrained model
=> loading layer5.0.downsample.1.bias from pretrained model
=> loading layer5.0.downsample.1.running_mean from pretrained model
=> loading layer5.0.downsample.1.running_var from pretrained model
=> loading layer5.0.downsample.1.num_batches_tracked from pretrained model
=> loading layer5.1.conv1.weight from pretrained model
=> loading layer5.1.bn1.weight from pretrained model
=> loading layer5.1.bn1.bias from pretrained model
=> loading layer5.1.bn1.running_mean from pretrained model
=> loading layer5.1.bn1.running_var from pretrained model
=> loading layer5.1.bn1.num_batches_tracked from pretrained model
=> loading layer5.1.conv2.weight from pretrained model
=> loading layer5.1.bn2.weight from pretrained model
=> loading layer5.1.bn2.bias from pretrained model
=> loading layer5.1.bn2.running_mean from pretrained model
=> loading layer5.1.bn2.running_var from pretrained model
=> loading layer5.1.bn2.num_batches_tracked from pretrained model
=> loading layer5.1.conv3.weight from pretrained model
=> loading layer5.1.bn3.weight from pretrained model
=> loading layer5.1.bn3.bias from pretrained model
=> loading layer5.1.bn3.running_mean from pretrained model
=> loading layer5.1.bn3.running_var from pretrained model
=> loading layer5.1.bn3.num_batches_tracked from pretrained model
=> loading compression3.0.weight from pretrained model
=> loading compression3.1.weight from pretrained model
=> loading compression3.1.bias from pretrained model
=> loading compression3.1.running_mean from pretrained model
=> loading compression3.1.running_var from pretrained model
=> loading compression3.1.num_batches_tracked from pretrained model
=> loading compression4.0.weight from pretrained model
=> loading compression4.1.weight from pretrained model
=> loading compression4.1.bias from pretrained model
=> loading compression4.1.running_mean from pretrained model
=> loading compression4.1.running_var from pretrained model
=> loading compression4.1.num_batches_tracked from pretrained model
=> loading pag3.f_x.0.weight from pretrained model
=> loading pag3.f_x.1.weight from pretrained model
=> loading pag3.f_x.1.bias from pretrained model
=> loading pag3.f_x.1.running_mean from pretrained model
=> loading pag3.f_x.1.running_var from pretrained model
=> loading pag3.f_x.1.num_batches_tracked from pretrained model
=> loading pag3.f_y.0.weight from pretrained model
=> loading pag3.f_y.1.weight from pretrained model
=> loading pag3.f_y.1.bias from pretrained model
=> loading pag3.f_y.1.running_mean from pretrained model
=> loading pag3.f_y.1.running_var from pretrained model
=> loading pag3.f_y.1.num_batches_tracked from pretrained model
=> loading pag4.f_x.0.weight from pretrained model
=> loading pag4.f_x.1.weight from pretrained model
=> loading pag4.f_x.1.bias from pretrained model
=> loading pag4.f_x.1.running_mean from pretrained model
=> loading pag4.f_x.1.running_var from pretrained model
=> loading pag4.f_x.1.num_batches_tracked from pretrained model
=> loading pag4.f_y.0.weight from pretrained model
=> loading pag4.f_y.1.weight from pretrained model
=> loading pag4.f_y.1.bias from pretrained model
=> loading pag4.f_y.1.running_mean from pretrained model
=> loading pag4.f_y.1.running_var from pretrained model
=> loading pag4.f_y.1.num_batches_tracked from pretrained model
=> loading layer3_.0.conv1.weight from pretrained model
=> loading layer3_.0.bn1.weight from pretrained model
=> loading layer3_.0.bn1.bias from pretrained model
=> loading layer3_.0.bn1.running_mean from pretrained model
=> loading layer3_.0.bn1.running_var from pretrained model
=> loading layer3_.0.bn1.num_batches_tracked from pretrained model
=> loading layer3_.0.conv2.weight from pretrained model
=> loading layer3_.0.bn2.weight from pretrained model
=> loading layer3_.0.bn2.bias from pretrained model
=> loading layer3_.0.bn2.running_mean from pretrained model
=> loading layer3_.0.bn2.running_var from pretrained model
=> loading layer3_.0.bn2.num_batches_tracked from pretrained model
=> loading layer3_.1.conv1.weight from pretrained model
=> loading layer3_.1.bn1.weight from pretrained model
=> loading layer3_.1.bn1.bias from pretrained model
=> loading layer3_.1.bn1.running_mean from pretrained model
=> loading layer3_.1.bn1.running_var from pretrained model
=> loading layer3_.1.bn1.num_batches_tracked from pretrained model
=> loading layer3_.1.conv2.weight from pretrained model
=> loading layer3_.1.bn2.weight from pretrained model
=> loading layer3_.1.bn2.bias from pretrained model
=> loading layer3_.1.bn2.running_mean from pretrained model
=> loading layer3_.1.bn2.running_var from pretrained model
=> loading layer3_.1.bn2.num_batches_tracked from pretrained model
=> loading layer4_.0.conv1.weight from pretrained model
=> loading layer4_.0.bn1.weight from pretrained model
=> loading layer4_.0.bn1.bias from pretrained model
=> loading layer4_.0.bn1.running_mean from pretrained model
=> loading layer4_.0.bn1.running_var from pretrained model
=> loading layer4_.0.bn1.num_batches_tracked from pretrained model
=> loading layer4_.0.conv2.weight from pretrained model
=> loading layer4_.0.bn2.weight from pretrained model
=> loading layer4_.0.bn2.bias from pretrained model
=> loading layer4_.0.bn2.running_mean from pretrained model
=> loading layer4_.0.bn2.running_var from pretrained model
=> loading layer4_.0.bn2.num_batches_tracked from pretrained model
=> loading layer4_.1.conv1.weight from pretrained model
=> loading layer4_.1.bn1.weight from pretrained model
=> loading layer4_.1.bn1.bias from pretrained model
=> loading layer4_.1.bn1.running_mean from pretrained model
=> loading layer4_.1.bn1.running_var from pretrained model
=> loading layer4_.1.bn1.num_batches_tracked from pretrained model
=> loading layer4_.1.conv2.weight from pretrained model
=> loading layer4_.1.bn2.weight from pretrained model
=> loading layer4_.1.bn2.bias from pretrained model
=> loading layer4_.1.bn2.running_mean from pretrained model
=> loading layer4_.1.bn2.running_var from pretrained model
=> loading layer4_.1.bn2.num_batches_tracked from pretrained model
=> loading layer5_.0.conv1.weight from pretrained model
=> loading layer5_.0.bn1.weight from pretrained model
=> loading layer5_.0.bn1.bias from pretrained model
=> loading layer5_.0.bn1.running_mean from pretrained model
=> loading layer5_.0.bn1.running_var from pretrained model
=> loading layer5_.0.bn1.num_batches_tracked from pretrained model
=> loading layer5_.0.conv2.weight from pretrained model
=> loading layer5_.0.bn2.weight from pretrained model
=> loading layer5_.0.bn2.bias from pretrained model
=> loading layer5_.0.bn2.running_mean from pretrained model
=> loading layer5_.0.bn2.running_var from pretrained model
=> loading layer5_.0.bn2.num_batches_tracked from pretrained model
=> loading layer5_.0.conv3.weight from pretrained model
=> loading layer5_.0.bn3.weight from pretrained model
=> loading layer5_.0.bn3.bias from pretrained model
=> loading layer5_.0.bn3.running_mean from pretrained model
=> loading layer5_.0.bn3.running_var from pretrained model
=> loading layer5_.0.bn3.num_batches_tracked from pretrained model
=> loading layer5_.0.downsample.0.weight from pretrained model
=> loading layer5_.0.downsample.1.weight from pretrained model
=> loading layer5_.0.downsample.1.bias from pretrained model
=> loading layer5_.0.downsample.1.running_mean from pretrained model
=> loading layer5_.0.downsample.1.running_var from pretrained model
=> loading layer5_.0.downsample.1.num_batches_tracked from pretrained model
=> loading layer3_d.conv1.weight from pretrained model
=> loading layer3_d.bn1.weight from pretrained model
=> loading layer3_d.bn1.bias from pretrained model
=> loading layer3_d.bn1.running_mean from pretrained model
=> loading layer3_d.bn1.running_var from pretrained model
=> loading layer3_d.bn1.num_batches_tracked from pretrained model
=> loading layer3_d.conv2.weight from pretrained model
=> loading layer3_d.bn2.weight from pretrained model
=> loading layer3_d.bn2.bias from pretrained model
=> loading layer3_d.bn2.running_mean from pretrained model
=> loading layer3_d.bn2.running_var from pretrained model
=> loading layer3_d.bn2.num_batches_tracked from pretrained model
=> loading layer3_d.downsample.0.weight from pretrained model
=> loading layer3_d.downsample.1.weight from pretrained model
=> loading layer3_d.downsample.1.bias from pretrained model
=> loading layer3_d.downsample.1.running_mean from pretrained model
=> loading layer3_d.downsample.1.running_var from pretrained model
=> loading layer3_d.downsample.1.num_batches_tracked from pretrained model
=> loading layer4_d.0.conv1.weight from pretrained model
=> loading layer4_d.0.bn1.weight from pretrained model
=> loading layer4_d.0.bn1.bias from pretrained model
=> loading layer4_d.0.bn1.running_mean from pretrained model
=> loading layer4_d.0.bn1.running_var from pretrained model
=> loading layer4_d.0.bn1.num_batches_tracked from pretrained model
=> loading layer4_d.0.conv2.weight from pretrained model
=> loading layer4_d.0.bn2.weight from pretrained model
=> loading layer4_d.0.bn2.bias from pretrained model
=> loading layer4_d.0.bn2.running_mean from pretrained model
=> loading layer4_d.0.bn2.running_var from pretrained model
=> loading layer4_d.0.bn2.num_batches_tracked from pretrained model
=> loading layer4_d.0.conv3.weight from pretrained model
=> loading layer4_d.0.bn3.weight from pretrained model
=> loading layer4_d.0.bn3.bias from pretrained model
=> loading layer4_d.0.bn3.running_mean from pretrained model
=> loading layer4_d.0.bn3.running_var from pretrained model
=> loading layer4_d.0.bn3.num_batches_tracked from pretrained model
=> loading layer4_d.0.downsample.0.weight from pretrained model
=> loading layer4_d.0.downsample.1.weight from pretrained model
=> loading layer4_d.0.downsample.1.bias from pretrained model
=> loading layer4_d.0.downsample.1.running_mean from pretrained model
=> loading layer4_d.0.downsample.1.running_var from pretrained model
=> loading layer4_d.0.downsample.1.num_batches_tracked from pretrained model
=> loading diff3.0.weight from pretrained model
=> loading diff3.1.weight from pretrained model
=> loading diff3.1.bias from pretrained model
=> loading diff3.1.running_mean from pretrained model
=> loading diff3.1.running_var from pretrained model
=> loading diff3.1.num_batches_tracked from pretrained model
=> loading diff4.0.weight from pretrained model
=> loading diff4.1.weight from pretrained model
=> loading diff4.1.bias from pretrained model
=> loading diff4.1.running_mean from pretrained model
=> loading diff4.1.running_var from pretrained model
=> loading diff4.1.num_batches_tracked from pretrained model
=> loading spp.scale1.1.weight from pretrained model
=> loading spp.scale1.1.bias from pretrained model
=> loading spp.scale1.1.running_mean from pretrained model
=> loading spp.scale1.1.running_var from pretrained model
=> loading spp.scale1.1.num_batches_tracked from pretrained model
=> loading spp.scale1.3.weight from pretrained model
=> loading spp.scale2.1.weight from pretrained model
=> loading spp.scale2.1.bias from pretrained model
=> loading spp.scale2.1.running_mean from pretrained model
=> loading spp.scale2.1.running_var from pretrained model
=> loading spp.scale2.1.num_batches_tracked from pretrained model
=> loading spp.scale2.3.weight from pretrained model
=> loading spp.scale3.1.weight from pretrained model
=> loading spp.scale3.1.bias from pretrained model
=> loading spp.scale3.1.running_mean from pretrained model
=> loading spp.scale3.1.running_var from pretrained model
=> loading spp.scale3.1.num_batches_tracked from pretrained model
=> loading spp.scale3.3.weight from pretrained model
=> loading spp.scale4.1.weight from pretrained model
=> loading spp.scale4.1.bias from pretrained model
=> loading spp.scale4.1.running_mean from pretrained model
=> loading spp.scale4.1.running_var from pretrained model
=> loading spp.scale4.1.num_batches_tracked from pretrained model
=> loading spp.scale4.3.weight from pretrained model
=> loading spp.scale0.0.weight from pretrained model
=> loading spp.scale0.0.bias from pretrained model
=> loading spp.scale0.0.running_mean from pretrained model
=> loading spp.scale0.0.running_var from pretrained model
=> loading spp.scale0.0.num_batches_tracked from pretrained model
=> loading spp.scale0.2.weight from pretrained model
=> loading spp.scale_process.0.weight from pretrained model
=> loading spp.scale_process.0.bias from pretrained model
=> loading spp.scale_process.0.running_mean from pretrained model
=> loading spp.scale_process.0.running_var from pretrained model
=> loading spp.scale_process.0.num_batches_tracked from pretrained model
=> loading spp.scale_process.2.weight from pretrained model
=> loading spp.compression.0.weight from pretrained model
=> loading spp.compression.0.bias from pretrained model
=> loading spp.compression.0.running_mean from pretrained model
=> loading spp.compression.0.running_var from pretrained model
=> loading spp.compression.0.num_batches_tracked from pretrained model
=> loading spp.compression.2.weight from pretrained model
=> loading spp.shortcut.0.weight from pretrained model
=> loading spp.shortcut.0.bias from pretrained model
=> loading spp.shortcut.0.running_mean from pretrained model
=> loading spp.shortcut.0.running_var from pretrained model
=> loading spp.shortcut.0.num_batches_tracked from pretrained model
=> loading spp.shortcut.2.weight from pretrained model
=> loading dfm.conv_p.0.weight from pretrained model
=> loading dfm.conv_p.1.weight from pretrained model
=> loading dfm.conv_p.1.bias from pretrained model
=> loading dfm.conv_p.1.running_mean from pretrained model
=> loading dfm.conv_p.1.running_var from pretrained model
=> loading dfm.conv_p.1.num_batches_tracked from pretrained model
=> loading dfm.conv_i.0.weight from pretrained model
=> loading dfm.conv_i.1.weight from pretrained model
=> loading dfm.conv_i.1.bias from pretrained model
=> loading dfm.conv_i.1.running_mean from pretrained model
=> loading dfm.conv_i.1.running_var from pretrained model
=> loading dfm.conv_i.1.num_batches_tracked from pretrained model
=> loading layer5_d.0.conv1.weight from pretrained model
=> loading layer5_d.0.bn1.weight from pretrained model
=> loading layer5_d.0.bn1.bias from pretrained model
=> loading layer5_d.0.bn1.running_mean from pretrained model
=> loading layer5_d.0.bn1.running_var from pretrained model
=> loading layer5_d.0.bn1.num_batches_tracked from pretrained model
=> loading layer5_d.0.conv2.weight from pretrained model
=> loading layer5_d.0.bn2.weight from pretrained model
=> loading layer5_d.0.bn2.bias from pretrained model
=> loading layer5_d.0.bn2.running_mean from pretrained model
=> loading layer5_d.0.bn2.running_var from pretrained model
=> loading layer5_d.0.bn2.num_batches_tracked from pretrained model
=> loading layer5_d.0.conv3.weight from pretrained model
=> loading layer5_d.0.bn3.weight from pretrained model
=> loading layer5_d.0.bn3.bias from pretrained model
=> loading layer5_d.0.bn3.running_mean from pretrained model
=> loading layer5_d.0.bn3.running_var from pretrained model
=> loading layer5_d.0.bn3.num_batches_tracked from pretrained model
=> loading layer5_d.0.downsample.0.weight from pretrained model
=> loading layer5_d.0.downsample.1.weight from pretrained model
=> loading layer5_d.0.downsample.1.bias from pretrained model
=> loading layer5_d.0.downsample.1.running_mean from pretrained model
=> loading layer5_d.0.downsample.1.running_var from pretrained model
=> loading layer5_d.0.downsample.1.num_batches_tracked from pretrained model
=> loading seghead_p.bn1.weight from pretrained model
=> loading seghead_p.bn1.bias from pretrained model
=> loading seghead_p.bn1.running_mean from pretrained model
=> loading seghead_p.bn1.running_var from pretrained model
=> loading seghead_p.bn1.num_batches_tracked from pretrained model
=> loading seghead_p.conv1.weight from pretrained model
=> loading seghead_p.bn2.weight from pretrained model
=> loading seghead_p.bn2.bias from pretrained model
=> loading seghead_p.bn2.running_mean from pretrained model
=> loading seghead_p.bn2.running_var from pretrained model
=> loading seghead_p.bn2.num_batches_tracked from pretrained model
=> loading seghead_p.conv2.weight from pretrained model
=> loading seghead_p.conv2.bias from pretrained model
=> loading seghead_d.bn1.weight from pretrained model
=> loading seghead_d.bn1.bias from pretrained model
=> loading seghead_d.bn1.running_mean from pretrained model
=> loading seghead_d.bn1.running_var from pretrained model
=> loading seghead_d.bn1.num_batches_tracked from pretrained model
=> loading seghead_d.conv1.weight from pretrained model
=> loading seghead_d.bn2.weight from pretrained model
=> loading seghead_d.bn2.bias from pretrained model
=> loading seghead_d.bn2.running_mean from pretrained model
=> loading seghead_d.bn2.running_var from pretrained model
=> loading seghead_d.bn2.num_batches_tracked from pretrained model
=> loading seghead_d.conv2.weight from pretrained model
=> loading seghead_d.conv2.bias from pretrained model
=> loading final_layer.bn1.weight from pretrained model
=> loading final_layer.bn1.bias from pretrained model
=> loading final_layer.bn1.running_mean from pretrained model
=> loading final_layer.bn1.running_var from pretrained model
=> loading final_layer.bn1.num_batches_tracked from pretrained model
=> loading final_layer.conv1.weight from pretrained model
=> loading final_layer.bn2.weight from pretrained model
=> loading final_layer.bn2.bias from pretrained model
=> loading final_layer.bn2.running_mean from pretrained model
=> loading final_layer.bn2.running_var from pretrained model
=> loading final_layer.bn2.num_batches_tracked from pretrained model
=> loading final_layer.conv2.weight from pretrained model
=> loading final_layer.conv2.bias from pretrained model
  0% 0/140 [00:00<?, ?it/s]

Colab paid products - Cancel contracts here
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
Removed secondary cursors
