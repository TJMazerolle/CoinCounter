# Inputs ###########################################################################################################
model_name = "CoinDetectionModel"       # model assumed to be in ./Models
video_to_annotate = "UnAnnotatedCoins"  # video assumed to be in ./Videos/UnannotatedVideos
annotated_video_name = "AnnotatedCoins" # video assumed to be saved in ./Videos/AnnotatedVideos

class_value_dict = {'nickel': 0.05, 'dime': 0.1, 'quarter': 0.25,
                    'loonie': 1.0, 'toonie': 2.0}
####################################################################################################################

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../HelperScripts')

from AnnotateVideo import create_annotated_video
from ultralytics import YOLO

# Directories
model_file = f'../../Models/{model_name}.pt'
unannotated_video_directory = f"../../Videos/UnannotatedVideos/{video_to_annotate}.mp4"
annotated_video_directory = f"../../Videos/AnnotatedVideos/{annotated_video_name}.mp4"

model = YOLO(model_file)
create_annotated_video(unannotated_video_directory, annotated_video_directory, 
                       model, class_value_dict, prediction_confidence = 0.7, thickness=5)