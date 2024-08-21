import os
import shutil
import torch
from ultralytics import YOLO

def check_GPU():
    if torch.cuda.is_available():
        print("GPU is available!")
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("GPU is not available, using CPU instead.")
 
def train_YOLO_model(
        # default values based off those found in https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings
        training_data_yaml, model = "yolov8n.yaml", epochs = 100,
        time = None, patience = 100, batch = 16, imgsz = 640, save = True,
        save_period = -1, cache = False, device = None, workers = 8, project = None,
        name = None, exist_ok = False, pretrained = True, optimizer = 'auto', 
        verbose = False, seed = 0, deterministic = True, single_cls = False,
        rect = False, cos_lr = False, close_mosaic = 10, resume = False, amp = True,
        fraction = 1.0, profile = False, freeze = None, lr0 = 0.01, lrf = 0.01, 
        momentum = 0.937, weight_decay = 0.0005, warmup_epochs = 3.0, 
        warmup_momentum = 0.8, warmup_bias_lr = 0.1, box = 7.5, cls = 0.5,
        dfl = 1.5, pose = 12.0, kobj = 2.0, label_smoothing = 0.0, nbs = 64,
        overlap_mask = True, mask_ratio = 4, dropout = 0.0, val = True, plots = False):
    model = YOLO(model)  # build a new model from scratch
    model.train(
        data = training_data_yaml, epochs = epochs,
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
    return model

def move_created_model(new_filename, destination_directory = "../../Models"):
    train_directory = os.listdir("./runs/detect")
    current_train_folder = train_directory[len(train_directory) - 1]
    model_location = f"./runs/detect/{current_train_folder}/weights/last.pt"
    destination_path = os.path.join(destination_directory, new_filename)
    shutil.move(model_location, destination_path)

