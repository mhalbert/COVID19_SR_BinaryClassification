import cv2
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

#import SR inferencing
import test

sns.set(style = "darkgrid")
SEED = 12

def data_constructor(filepath, classes , dim_size ,index  ,bboxes , interpolation = cv2.INTER_AREA):
    """Constructs and splits X and Y for training , validtion and test"""
    np.random.seed(SEED)
    y = np.array(classes[index])
    x = []

    for i in index:
        img = cv2.imread(filepath[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x1,y1,x2,y2 = bboxes[i]
        img = img[y1:y2,x1:x2]
        img64 = cv2.resize(img, dim_size , interpolation = interpolation)
        img128 = cv2.resize(img, (128, 128) , interpolation = interpolation)
        filename, _ = os.path.splitext(os.path.basename(filepath[i]))
        #print(filename)
        cv2.imwrite('64res/' + filename + '_64.png', img64)
        cv2.imwrite('128res/' + filename + '_128.png', img128)


    # Run sr step on 64res
    test.inference('64res/')
    # loop through SR output folder /results/SR/MyImage/FAWDN/
    for filename in os.listdir('/kaggle/working/MyImage/FAWDN/x2'):
        img=cv2.imread(os.path.join('/kaggle/working/MyImage/FAWDN/x2', filename))
        # img open then grab the image data then append that
        x.append(img)
    x = np.array(x)

    print("==================================================")
    print("Successfully created dataset. Ready for classificaiton.")
    print("==================================================")

    return x, y

# Auxillary data prep functions
def load_labels(label_file):
    """Loads image filenames, classes, and bounding boxes"""
    fnames, classes, bboxes = [], [], []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            fname, cls, xmin, ymin, xmax, ymax = line.strip('\n').split()
            fnames.append(fname)
            classes.append(int(cls))
            bboxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
    fnames = ["/kaggle/input/covidxct/3A_images/" + fname for fname in fnames]
    fnames = np.array(fnames)
    classes = np.array(classes)
    return fnames, classes, bboxes

def index_generator(fnames , SET):
    """Genrated random index of a particular class"""
    np.random.seed(SEED)
    index = np.random.randint(1,len(fnames),size = SET)
    return index

IMG_HEIGHT = 64
IMG_WIDTH = 64
DIM = (IMG_HEIGHT, IMG_WIDTH)
VALID_SET= 50

label_file_train = "train_COVIDx-CT.txt"
label_file_valid = "/kaggle/input/covidxct/val_COVIDx_CT-3A.txt"

fnames_valid, classes_valid, bboxes_valid = load_labels(label_file_valid)
print(fnames_valid)
valid_index = index_generator(fnames_valid, VALID_SET)

x_valid , y_valid = data_constructor(fnames_valid, classes_valid, DIM, index=valid_index, bboxes = bboxes_valid)
x_valid = tf.keras.applications.densenet.preprocess_input(x_valid)

# import pretrained binary models
print("===>Loading Pre-trained Models for Phase 1 & 2")
modelPhase1 = tf.keras.models.load_model('/kaggle/input/pretrained-models/BinaryPhase1BaseRun.h5')
modelPhase2 = tf.keras.models.load_model('/kaggle/input/pretrained-models/BinaryPhase2NormalCap.h5')
# inference on x_valid
print("===>Phase 1 Inferencing")
y_pred1  = modelPhase1.predict(x_valid)
print("Successfully Classified Covid.")
print("==================================================")
x_valid_noncovid = np.where(y_pred1 < 0.5, x_valid, np.zeros_like(x_valid))

#pass filtered normal/cap to phase 2
print("Phase 2 inferencing")
y_pred2 = modelPhase2.predict(x_valid_noncovid)
print("Successfully Classified CAP.")
print("Successfully Classified Normal.")
print("==================================================")

#0 normal, 1 pnemnia, 2 covid
y_pred_final = np.where(y_pred1 > 0.5, 2, np.where(y_pred2 > 0.5, 1, 0))
acc = accuracy_score(y_valid, y_pred_final)
print(acc)
