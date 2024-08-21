# DiceCounter
This repository provides everything to train and test a dice counting model.

I run these scripts using a Python 3.9 environment.  The required libraries can be found in the requirements.txt file.  Note that in order to run the scripts on a GPU, you will need to install additional dependencies.  Go to https://pytorch.org/get-started/locally/ and run the suggested command for your needs.

DiceCounterTrainingScript.py will train the model using the training and validation image sets in the ImageSet directory.  VideoAnnotator.py will look for a given video file name in the Videos directory and annotate each frame based on the model built by DiceCounterTrainingScript.py.  The resulting annotated video will be stored in the Videos directory.