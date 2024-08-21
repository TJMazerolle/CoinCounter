import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../HelperScripts')
from YOLOModelTraining import *

check_GPU()

if __name__ == '__main__':
    model = train_YOLO_model(training_data_yaml = "../../ImageSets/YAMLFormat/Coins/data.yaml", training_epochs = 1000)
    move_created_model("CoinDetectionModel.pt")