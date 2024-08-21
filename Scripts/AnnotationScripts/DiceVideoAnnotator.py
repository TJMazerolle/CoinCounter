import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('./HelperScripts')

from AnnotateVideo import create_annotated_video
from ultralytics import YOLO

class_value_dict = {'one': 1, 'two': 2, 'three': 3,
                    'four': 4, 'five':5, 'six':6}
model = YOLO('./Models/DiceDetectionModel.pt')
create_annotated_video("./Videos/UnAnnotatedDice.mp4", "./Videos/AnnotatedDice.mp4", model, class_value_dict, prediction_confidence = 0.7,
                       thickness=5)