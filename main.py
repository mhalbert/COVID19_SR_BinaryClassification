import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import seaborn as sns

sns.set(style = "darkgrid")
SEED = 12


def data_constructor(filepath, classes , dim_size ,index  ,bboxes , interpolation = cv2.INTER_AREA , intensify =False):
    """Constructs and splits X and Y for training , validtion and test"""
    np.random.seed(SEED)
    y = np.array(classes[index])
    x = []
    for i in index:
        img  = cv2.imread(filepath[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x1,y1,x2,y2 = bboxes[i]
        img = img[y1:y2,x1:x2]
        img64 = cv2.resize(img, dim_size , interpolation = interpolation)
        img128 = cv2.resize(img, (128, 128) , interpolation = interpolation)
        cv2.imwrite('/src/64res/ct' + i + '_64.png', img64)
        cv2.imwrite('/src/128res/ct' + i + '_128.png', img128)
    #Run sr step on 64res and 128res folder
    # loop through SR output folder new128res
        x.append(img)

    x = np.array(x)
    if intensify == True:
        x= x/255
    return x , y

IMG_HEIGHT = 64
IMG_WIDTH = 64
DIM = (IMG_HEIGHT, IMG_WIDTH)
TRAIN_SET= 46778
VALID_SET= 6486
EPOCHS = 40
BS = 32
n = 8000
LR = 0.0001
label_file_train = "src/covidxct/train_COVIDx_CT-3A.txt"
label_file_valid = "src/covidxct/val_COVIDx_CT-3A.txt"

fnames_train, classes_train, bboxes_train = load_labels(label_file_train)
fnames_valid, classes_valid, bboxes_valid = load_labels(label_file_valid)
train_index = index_generator(fnames_train, TRAIN_SET)
valid_index = index_generator(fnames_valid, VALID_SET)
train_index_updated = train_index_updater(classes_train,train_index, n)
df = dataframe_generator(train_index_updated, valid_index, classes_train, classes_valid)
df.plot.bar(title = "Image Distribution");

x_valid , y_valid= data_constructor(fnames_valid, classes_valid, DIM, index=valid_index, bboxes = bboxes_valid)

x_valid = tf.keras.applications.densenet.preprocess_input(x_valid)

for i in range(len(y_valid)):
    if y_valid[i] ==1:
        y_valid[i]=0
    if y_valid[i]==2:
        y_valid[i]=1

# import pretrained binary models
# inference on x_valid

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

def numberofclasses(classes, index):
    class0 = len((np.where(classes[index]==0))[0])
    class1 = len((np.where(classes[index]==1))[0])
    class2 = len((np.where(classes[index]==2))[0])
    return class0  , class1, class2
def dataframe_generator(train_index , valid_index , classes_train , classes_valid ):
    """Returns 1 dataframes of datasets distribution"""
    index = ["Normal" , "Pneumonia" , "COIVD -19"]
    train_DF = numberofclasses(classes_train, train_index)
    valid_DF = numberofclasses(classes_valid, valid_index)
    df = pd.DataFrame({'train': train_DF ,'valid' : valid_DF} , index = index)
    return df
def train_index_updater(classes_train , train_index,n ):
    """Updates train_index for class balance"""
    np.random.seed(SEED)
    class0_train = np.where(classes_train[train_index]==0)[0]
    class1_train = np.where(classes_train[train_index]==1)[0]
    class2_train = np.where(classes_train[train_index]==2)[0]
    class0 =train_index[class0_train]
    class1 = train_index[class1_train]
    np.random.seed(SEED)
    class22 = np.random.choice(class2_train , n)
    class2 = train_index[class22]
    train_index_updated = np.concatenate((class0 , class1 , class2))
    np.random.shuffle(train_index_updated)
    return train_index_updated
