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

def train_YOLO_model(training_epochs = 1000, yolo_model = "yolov8n.yaml", training_data_yaml = "./ImageSet/data.yaml"):
    model = YOLO(yolo_model)  # build a new model from scratch
    model.train(data = training_data_yaml, epochs = training_epochs)
    return model

def move_created_model(new_filename, destination_directory = "./Models"):
    train_directory = os.listdir("./runs/detect")
    current_train_folder = train_directory[len(train_directory) - 1]
    model_location = f"./runs/detect/{current_train_folder}/weights/last.pt"
    destination_path = os.path.join(destination_directory, new_filename)
    shutil.move(model_location, destination_path)

