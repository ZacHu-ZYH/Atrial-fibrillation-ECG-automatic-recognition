![alt text](https://d20vrrgs8k4bvw.cloudfront.net/images/courses/logos/logo-color-tensorflow.png)

## Install WFDB Software Package
     Follow instructions in: https://physionet.org/physiotools/wfdb-linux-quick-start.shtml 

## Install requirements
    > pip install -r requirements.txt
    > apt-get install python-tk

## Install requirements
    > pip install -r requirements.txt

## Download Dataset
    > wget -r -np http://www.physionet.org/physiobank/database/afpdb/

## Copy paste afpdb folder to current directory 
    > mv www.physionet.org/physiobank/database/afpdb .

## Go to src folder
    > cd src

## Create CSV files 
    > python dat2csv.py

## Create training and test dataset
    > mkdir training test
    
## Create .npy files 
    >  python create_dataset.py

## Create Neural Network model and train it 
    > python train.py

## Best Model
     Rename the best model from my_model.h5 to best.h5
     

## Make predictions with best model (90% accuracy)
    > python predict.py
## Tensoarboard
    > tensorboard --logdir=logs/
## Accuracy & Loss
![alt text](img/90.png)

## Normal Record
![alt text](img/Normal.png)
## Abnormal Record
![alt text](img/Abnormal.png)


