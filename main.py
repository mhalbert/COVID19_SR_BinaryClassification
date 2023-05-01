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
from sklearn.metrics import confusion_matrix
from itertools import product
import matplotlib.image as mpimg

#import SR inferencing
import test

sns.set(style = "darkgrid")
SEED = 12

def data_constructor(filepath, classes , dim_size ,index  ,bboxes , interpolation = cv2.INTER_AREA):
    """Constructs and splits X and Y for training , validtion and test"""
    normal = "_0"
    cap = "_1"
    covid = "_2"

    #np.random.seed(SEED) ?
    y = np.array(classes[index])
    #print(index, y)

    # we can clean up these paths if we want
    if os.path.exists('/kaggle/working/64res/'):
        shutil.rmtree('/kaggle/working/64res/')
        os.makedirs('/kaggle/working/64res/')
    else:
        os.makedirs('/kaggle/working/64res/')

    if os.path.exists('/kaggle/working/128res/'):
        shutil.rmtree('/kaggle/working/128res/')
        os.makedirs('/kaggle/working/128res/')
    else:
        os.makedirs('/kaggle/working/128res/')

    if os.path.exists('/kaggle/working/MyImage/FAWDN/x2'):
        shutil.rmtree('/kaggle/working/MyImage/FAWDN/x2')
        os.makedirs('/kaggle/working/MyImage/FAWDN/x2')

    x = []
    label_index = 0
    for i in index:
        img = cv2.imread(filepath[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x1,y1,x2,y2 = bboxes[i]
        img = img[y1:y2,x1:x2]
        img64 = cv2.resize(img, dim_size , interpolation = interpolation)
        img128 = cv2.resize(img, (128,128), interpolation = interpolation)
        filename, _ = os.path.splitext(os.path.basename(filepath[i]))
        #print(y[label_index])
        if y[label_index] == 0:
            class_ext = normal
        elif y[label_index] == 1:
            class_ext = cap
        else:
            class_ext = covid

        label_index += 1

        try:
            cv2.imwrite('/kaggle/working/64res/' + filename + '_64' + class_ext + '.png', img64)
        except:
            print("Error! Didn't write 64x64: ", filename )
        try:
            cv2.imwrite('/kaggle/working/128res/' + filename + '_128' + class_ext + '.png', img128)
        except:
            print("Error! Didn't write 128x128: ", filename )



    #count = 0
    # Iterate directory
    #for path in os.listdir('/kaggle/working/64res/'):
        # check if current path is a file
    #    if os.path.isfile(os.path.join('/kaggle/working/64res/', path)):
    #        count += 1
    #    else:
    #        print(path)
    #print('Files in 64res/:', count)

    # Run sr step on 64res
    test.inference('/kaggle/working/64res/')
    # loop through SR output folder /results/SR/MyImage/FAWDN/
    i=0
    tempLabels = []
    for filename in os.listdir('/kaggle/working/MyImage/FAWDN/x2'):
        img=cv2.imread(os.path.join('/kaggle/working/MyImage/FAWDN/x2', filename))
        # img open then grab the image data then append that
        x.append(img)
        classLabel = int(filename.split('.png')[0][-1])
        #print(filename, classLabel)
        tempLabels.append(classLabel)
        #print(y[i])
        i += 1
    x = np.array(x)
    # here take off the class ext and y.append() !!!!!

    print("==================================================")
    print("Successfully created dataset. Ready for classification.")
    print("==================================================")

    return x, tempLabels

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
VALID_SET= 500

label_file_train = "train_COVIDx-CT.txt"
label_file_valid = "/kaggle/input/covidxct/val_COVIDx_CT-3A.txt"

fnames_valid, classes_valid, bboxes_valid = load_labels(label_file_valid)
#print(len(fnames_valid))
valid_index = index_generator(fnames_valid, VALID_SET)
#print("Length of index generator:", len(valid_index))
x_valid , y_valid = data_constructor(fnames_valid, classes_valid, DIM, index=valid_index, bboxes = bboxes_valid)
x_valid = tf.keras.applications.densenet.preprocess_input(x_valid)

# import pretrained binary models
print("===> Loading Pre-trained Model for Phase 1")
modelPhase1 = tf.keras.models.load_model('/kaggle/input/pretrained-models/BinaryPhase1BaseRun.h5')
print("===> Loading Pre-trained Model for Phase 2")
modelPhase2 = tf.keras.models.load_model('/kaggle/input/pretrained-models/BinaryPhase2NormalCap.h5')
# inference on x_valid
print("===> Phase 1 Inferencing")
y_valid_array = np.array(y_valid)
y_pred1  = modelPhase1.predict(x_valid)
cm = confusion_matrix(y_valid_array, y_pred1)
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

# print(y_valid, y_pred_final)
# Compute avg accuracy score
acc = accuracy_score(y_valid, y_pred_final)

# Compute confusion matrix
cm = confusion_matrix(y_valid, y_pred_final)
# Define the labels for each class
class_names = ['Normal', 'Cap', 'Covid-19']

# Define the title of the confusion matrix
title = 'Confusion Matrix'

# Define the axis labels
xlabel = 'Predicted label'
ylabel = 'True label'

# Define the colors for the confusion matrix
cmap = 'Blues'

# Normalize the confusion matrix
normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Print the confusion matrix
print(title)
print()
print(np.array2string(cm, separator=', ',
                       formatter={'int': lambda x: f'{x:4d}'}))

# Print the normalized confusion matrix
print()
print('Normalized confusion matrix')
print(np.array2string(normalized_cm, separator=', ',
                       formatter={'float': lambda x: f'{x:5.2f}'}))

# Print the classification report
print()
print('Classification Report')
print('---------------------')
for i, class_name in enumerate(class_names):
    precision = cm[i,i] / cm[:,i].sum()
    recall = cm[i,i] / cm[i,:].sum()
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f'{class_name:<8} precision: {precision:.2f} recall: {recall:.2f} f1-score: {f1_score:.2f}')

print(acc)
