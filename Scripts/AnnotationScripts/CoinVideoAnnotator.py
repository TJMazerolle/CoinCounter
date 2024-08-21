import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('./HelperScripts')

from AnnotateVideo import create_annotated_video
from ultralytics import YOLO

class_value_dict = {'nickel': 0.05, 'dime': 0.1, 'quarter': 0.25,
                    'loonie': 1.0, 'toonie': 2.0}
model = YOLO('./Models/CoinDetectionModel.pt')
create_annotated_video("./Videos/UnAnnotatedCoins.mp4", "./Videos/AnnotatedCoins.mp4", model, class_value_dict, prediction_confidence = 0.7,
                       thickness=5)