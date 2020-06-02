import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np
import createmodel as crm
import pandas as pd
import ecg_plot as pl
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# ************************************* Tensorboard Settings ************************************************

NAME = "atrial"
tensorboard = TensorBoard(log_dir="../logs/{}".format(NAME))

# ************************************* Load Dataset ************************************************
x_data = np.load('../data/x_data.npy')
y_data = np.load('../data/y_data.npy')
train = 0.7
size = 200
mid = size / 2

x_train = np.array([])
x_train = np.append(x_train,x_data[:int(train * mid)])
x_train = np.append(x_train,x_data[mid:mid + int(train * mid)])
x_train = x_train.reshape(-1, 1280)

y_train = np.array([])
y_train = np.append(y_train,y_data[:int(train * mid)])
y_train = np.append(y_train,y_data[mid:mid + int(train * mid)])
y_train = y_train.astype(int)

x_test = np.array([])
x_test = np.append(x_test,x_data[int(train * mid):mid])
x_test = np.append(x_test,x_data[mid + int(train * mid):size])
x_test = x_test.reshape(-1, 1280)

y_test = np.array([])
y_test = np.append(y_test,y_data[int(train * mid):mid])
y_test = np.append(y_test,y_data[mid + int(train * mid):size])
y_test = y_test.astype(int)

print(x_data.shape)
print(y_data.shape)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train)
print(y_test)

# ************************************* Normalise data ***************************************************
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# ************************************* Create Model ***************************************************
learn_rate = 0.01 # Define learning rate
ep = 15 # Number of epochs
batch = 16 # define batch size
model = crm.create_model(learn_rate)
model.summary()
history = model.fit(x_train, y_train, epochs=ep,validation_data=(x_test, y_test), callbacks=[tensorboard])
# pd.DataFrame(history.history).to_csv(path_or_buf='logs/History.csv')

# ************************************* Model Accurancy ***************************************************
val_loss1, val_acc1 = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
print("Validation accuracy {:5.2f}%".format(100*val_acc1))
# print(val_acc1)  # model's accuracy
# print(val_loss1)  # model's loss (error)

# ************************************* Plot Accurancy & Loss ***************************************************
f, ax = plt.subplots(2, sharex=True)

ax[0].plot(history.history['acc'],'r',linewidth=2.0, label='Training Accuracy')
ax[0].plot(history.history['val_acc'],'b',linewidth=2.0, label='Testing Accuracy')
ax[0].legend(fontsize=10)
ax[0].set(xlabel='Epochs', ylabel='Accuracy')
ax[0].grid(b=True, which='major', color='#999999', linestyle='-',alpha=0.6)
ax[0].minorticks_on()
ax[0].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

ax[1].plot(history.history['loss'], 'g', linewidth=2.0, label='Training Loss')
ax[1].plot(history.history['val_loss'], 'y', linewidth=2.0, label='Testing Loss')
ax[1].legend(fontsize=10)
ax[1].set(xlabel='Epochs', ylabel='Loss Curves')
ax[1].grid(b=True, which='major', color='#999999', linestyle='-',alpha=0.6)
ax[1].minorticks_on()
ax[1].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.show()

# ************************************* Make predictions ***************************************************
predictions = model.predict(x_test)
test = 19
print("prediction:", np.argmax(predictions[test]))
print("real value:", y_test[test])

# Plot prediction
signal = x_test[test]
pl.ecg_plot(signal)

# ************************************* Metrics ***************************************************
y_pred = np.array([])
for x in range(0,len(predictions)):
	y_pred = np.append(y_pred,np.argmax(predictions[x]))
	
y_pred = y_pred.astype(int)

# Precision:
# The precision is the ratio tp / (tp + fp) where tp is the number of true positives and 
# fp the number of false positives. The precision is intuitively the ability of the classifier 
# not to label as positive a sample that is negative.
# The best value is 1 and the worst value is 0.

precision = precision_score(y_test, y_pred, average='micro') # micro calculates metrics globally by counting the total true positives, false negatives and false positives.
print("Precision:", precision)

# Recall (aka Sensitivity):
# Recall is the ratio of the correctly +ve labeled by our program to all who have atrial fibrillation in reality.
# Recall answers the following question: Of all the people who have atrial fibrillation, how many of those we correctly predict?
# Recall = TP/(TP+FN)
recall = recall_score(y_test, y_pred, average='micro') # micro calculates metrics globally by counting the total true positives, false negatives and false positives.
print("Recall:", recall) 

# Specificity:
# Specificity is the correctly -ve labeled by the program to all who are healthy in reality.
# Specifity answers the following question: Of all the people who are healthy, how many of those did we correctly predict?
# Specificity = TN/(TN+FP)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(tn, fp, fn, tp)
tn = float(tn)
specificity = tn / (tn + fp)
print("Specificity:", specificity) 

# More Details about metrics: https://towardsdatascience.com/accuracy-recall-precision-f-score-specificity-which-to-optimize-on-867d3f11124

# ************************************* Save Model ***************************************************
model.save('../models/my_model.h5') # Save entire model
# ======================================================================================================