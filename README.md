# SRCT COVID19 Classification 2023
### Project for CAP 5516 by Melody Halbert and Ethan Legum

The use of deep learning provides automated approaches to medical image analysis and is valuable for COVID-19 detection. 
This project focuses on the use of super-resolution to reconstruct high-quality CT images from under-sampled data. 
Our proposed method utilizes a prior super-resolution network to reconstruct high-resolution CT scans and a simple
two-stage binary classifier to classify COVID, pneumonia, and normal samples. By leveraging low-resolution scans, 
our approach aims to enable the use of lower-kilovoltage CT scans in practice while maintaining competitive 
classification accuracies.


![Our Architecture](architecture.png)

## Testing

### Method 1 (Recommended):
Running cloned git repo in a Kaggle notebook. Any personal changes to our scripts can 
be pushed to your forked repo and git pull-ed on Kaggle. 
```
   # Kaggle Cell Example:
   !git clone ./COVID19_SR_BinaryClassification.git
   !python /kaggle/working/COVID19_SR_BinaryClassification/main.py 
```

important paths for Kaggle usage (in main.py):
```
    path_64 = '/kaggle/working/64res/'
    path_128 = '/kaggle/working/128res/'
    fawdn_out = '/kaggle/working/MyImage/FAWDN/x2'
```

### Method 2:
Start running the SR and two-phase classification pipeline by running main.py. 
```
   $ python main.py
```
The critical portion of this method is ensuring your local pathing is correct. There is
many steps of data preprocessing and outputting into directories due to the SR step. 
important paths for local system usage (in main.py):
```
    # change these paths to resemble your working directory
    path_64 = '64res/'
    path_128 = '128res/'
    fawdn_out = 'MyImage/FAWDN/x2'
```



