import cv2
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shutil
from sklearn.metrics import accuracy_score
import matplotlib.image as mpimg

#import SR inferencing
import test

sns.set(style = "darkgrid")
SEED = 12

def data_constructor(filepath, classes , dim_size ,index  ,bboxes , interpolation = cv2.INTER_AREA):
    """Constructs and splits X and Y for training , validtion and test"""
    np.random.seed(SEED)
    y = np.array(classes[index])
    print(classes[index])
    print('Length of classes array: ', len(y))
    print('Length of index array: ', len(index))

    x = []
    for i in index:
        img = cv2.imread(filepath[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x1,y1,x2,y2 = bboxes[i]
        img = img[y1:y2,x1:x2]
        img64 = cv2.resize(img, dim_size , interpolation = interpolation)
        img128 = cv2.resize(img, (128,128), interpolation = interpolation)
        filename, _ = os.path.splitext(os.path.basename(filepath[i]))

        dir64 = '/kaggle/working/64res/'
        dir128 = '/kaggle/working/64res/'
        if os.path.exists(dir64):
            shutil.rmtree(dir64)
            os.makedirs(dir64)

        if os.path.exists(dir128):
            shutil.rmtree(dir128)
            os.makedirs(dir128)

        try:
            cv2.imwrite('/kaggle/working/64res/' + filename + '_64.png', img64)
        except:
            print("Error! Didn't write 64x64: ", filename )
        try:
            cv2.imwrite('/kaggle/working/64res/' + filename + '_128.png', img128)
        except:
            print("Error! Didn't write 128x128: ", filename )


    count = 0
    # Iterate directory
    for path in os.listdir(dir64):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir64, path)):
            count += 1
        else:
            print(path)
    print('Files in 64res/:', count)

    # Run sr step on 64res
    test.inference(dir64)
    # loop through SR output folder /results/SR/MyImage/FAWDN/
    i=0

    if os.path.exists('/kaggle/working/MyImage/FAWDN/x2'):
        shutil.rmtree('/kaggle/working/MyImage/FAWDN/x2')
        os.makedirs('/kaggle/working/MyImage/FAWDN/x2')

    for filename in os.listdir('/kaggle/working/MyImage/FAWDN/x2'):
        img=cv2.imread(os.path.join('/kaggle/working/MyImage/FAWDN/x2', filename))
        # img open then grab the image data then append that
        print(filename)
        x.append(img)
        print(y[i])
        i += 1
    x = np.array(x)

    print("==================================================")
    print("Successfully created dataset. Ready for classification.")
    print("==================================================")

    print(y)
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
    index = random.sample(range(len(fnames)), SET)
    return index

IMG_HEIGHT = 64
IMG_WIDTH = 64
DIM = (IMG_HEIGHT, IMG_WIDTH)
VALID_SET= 10

label_file_train = "train_COVIDx-CT.txt"
label_file_valid = "/kaggle/input/covidxct/val_COVIDx_CT-3A.txt"

fnames_valid, classes_valid, bboxes_valid = load_labels(label_file_valid)
#print(len(fnames_valid))
valid_index = index_generator(fnames_valid, VALID_SET)
print("Length of index generator:", len(valid_index))
x_valid , y_valid = data_constructor(fnames_valid, classes_valid, DIM, index=valid_index, bboxes = bboxes_valid)
x_valid = tf.keras.applications.densenet.preprocess_input(x_valid)

# import pretrained binary models
print("===> Loading Pre-trained Model for Phase 1")
modelPhase1 = tf.keras.models.load_model('/kaggle/input/pretrained-models/BinaryPhase1BaseRun.h5')
print("===> Loading Pre-trained Model for Phase 2")
modelPhase2 = tf.keras.models.load_model('/kaggle/input/pretrained-models/BinaryPhase2NormalCap.h5')
# inference on x_valid
print("===> Phase 1 Inferencing")
y_pred1  = modelPhase1.predict(x_valid)
print("Successfully Classified Covid.")
print("==================================================")
# mask values in x_valid that resulted in y_pred1 >= 0.5
mask = y_pred1 >= 0.5
mask_expanded = np.expand_dims(mask, axis=(1, 2))
x_valid_noncovid = np.where(mask_expanded, np.zeros_like(x_valid), x_valid)
#x_valid_noncovid = np.where(y_pred1 < 0.5, x_valid, np.zeros_like(x_valid) + np.expand_dims(np.array([0, 0, 0]), axis=0))

#pass filtered normal/cap to phase 2
print("===> Phase 2 inferencing")
y_pred2 = modelPhase2.predict(x_valid_noncovid)
print("Successfully Classified CAP.")
print("Successfully Classified Normal.")
print("==================================================")

#0 normal, 1 pnemnia, 2 covid
y_pred_final = np.where(y_pred1 > 0.5, 2, np.where(y_pred2 > 0.5, 1, 0))
#print(y_valid, y_pred_final)
acc = accuracy_score(y_valid, y_pred_final)
print(acc)





#
