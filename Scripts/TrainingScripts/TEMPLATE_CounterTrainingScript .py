# Inputs ###########################################################################################################

# In order to train the script as per your choices, make changes to the inputs in this section.
# Unless you are editing the code itself this should be the only place you should be making changes.

training_imageset_folder_name = "Coins"    # Change this to the desired folder name in ImageSets/YAMLFormat
desired_model_name = "CoinDetectionModel" # Change this to the name you want to give to the produced model

# Training Parameters
## Default values based off those found in https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings
model = "yolov8n.yaml" # see https://docs.ultralytics.com/models/#featured-models for the different model options
epochs = 1000
time = None 
patience = 100 
batch = 16 
imgsz = 640 
save = True
save_period = -1 
cache = False 
device = None 
workers = 8 
project = None
name = None 
exist_ok = False 
pretrained = True 
optimizer = 'auto' 
verbose = False 
seed = 0 
deterministic = True 
single_cls = False
rect = False 
cos_lr = False 
close_mosaic = 10 
resume = False 
amp = True
fraction = 1.0 
profile = False 
freeze = None 
lr0 = 0.01 
lrf = 0.01 
momentum = 0.937 
weight_decay = 0.0005 
warmup_epochs = 3.0 
warmup_momentum = 0.8 
warmup_bias_lr = 0.1 
box = 7.5 
cls = 0.5
dfl = 1.5 
pose = 12.0 
kobj = 2.0 
label_smoothing = 0.0 
nbs = 64
overlap_mask = True 
mask_ratio = 4 
dropout = 0.0 
val = True 
plots = False
####################################################################################################################

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../HelperScripts')
from YOLOModelTraining import *

check_GPU()

# Directories
training_image_data_directory = f"../../ImageSets/YAMLFormat/{training_imageset_folder_name}/data.yaml"
model_file = f"{desired_model_name}.pt"

if __name__ == '__main__':
    model = train_YOLO_model(training_data_yaml = training_image_data_directory, epochs = epochs,
        time = time, patience = patience, batch = batch, imgsz = imgsz, save = save,
        save_period = save_period, cache = cache, device = device, workers = workers, project = project,
        name = name, exist_ok = exist_ok, pretrained = pretrained, optimizer = optimizer, 
        verbose = verbose, seed = seed, deterministic = deterministic, single_cls = single_cls,
        rect = rect, cos_lr = cos_lr, close_mosaic = close_mosaic, resume = resume, amp = amp,
        fraction = fraction, profile = profile, freeze = freeze, lr0 = lr0, lrf = lrf, 
        momentum = momentum, weight_decay = weight_decay, warmup_epochs = warmup_epochs, 
        warmup_momentum = warmup_momentum, warmup_bias_lr = warmup_bias_lr, box = box, cls = cls,
        dfl = dfl, pose = pose, kobj = kobj, label_smoothing = label_smoothing, nbs = nbs,
        overlap_mask = overlap_mask, mask_ratio = mask_ratio, dropout = dropout, val = val, plots = plots)
    move_created_model(model_file) # will automatically save in ../../Models