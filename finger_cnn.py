# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 20:22:43 2020

@author: tharshi
"""

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, Callback


# load data
X = np.load('images.npy')

# input image dimensions
img_rows, img_cols = X.shape[1], X.shape[2]

# load labels
Y = np.load('labels.npy')

# define input shape for model
input_shape = (img_rows, img_cols, 3)

n_hands = X.shape[0]
#%%
# shuffle data
np.random.seed(27)

indices = np.arange(n_hands)
np.random.shuffle(indices)

X = X[indices, :, :, :]
Y = Y[indices, :]

#%%

# test split
test_split = 0.10
n_test = int(test_split * X.shape[0])

x_test = X[:n_test, :, :, :]
y_test = Y[:n_test, :]

x = X[n_test:, :, :, :]
y = Y[n_test:, :]


#%% Model
import os
from keras.layers import ELU

batch_size = 32
num_classes = 6
epochs = 1000
split = 0.30
pat = 300

model = Sequential()

model.add(Conv2D(8, kernel_size=(3, 3), 
                 input_shape=input_shape, 
                 name='input'))
model.add(ELU(name='ELU_1'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool_1', strides=(2,2)))
model.add(Dropout(0.25, name='dropout_1'))

model.add(Conv2D(8, kernel_size=(3, 3)))
model.add(ELU(name='ELU_2'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool_2', strides=(2,2)))
model.add(Dropout(0.25, name='dropout_2'))

model.add(Conv2D(8, kernel_size=(3, 3)))
model.add(ELU(name='ELU_3'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool_3', strides=(2,2)))
model.add(Dropout(0.25, name='dropout_3'))

model.add(Conv2D(8, kernel_size=(3, 3)))
model.add(ELU(name='ELU_4'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool_4', strides=(2,2)))
model.add(Dropout(0.25, name='dropout_4'))

model.add(Flatten())
model.add(Dense(512, activation='relu', name='feedforward'))
model.add(Dropout(0.50, name='dropout_dense'))
model.add(Dense(num_classes, activation='softmax', name='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adadelta',
              metrics=['accuracy'])

model.summary()

earlystop = EarlyStopping(verbose=True,
                          patience=pat,
                          monitor='val_loss')
if os.path.isfile('cnn_weights.h5'):
    print('Weights already exist~~')
    model.load_weights('cnn_weights.h5')
else:
    model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1, 
              shuffle=True, 
              validation_split=split,
              callbacks=[earlystop])
    model.save_weights('cnn_weights.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%% Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt

y_pred = model.predict(x_test, verbose=1)
con_mat = confusion_matrix(y_test.argmax(axis=1), 
                          y_pred.argmax(axis=1))

prec = np.diag(con_mat) / np.sum(con_mat, axis=0)
recall = np.diag(con_mat) / np.sum(con_mat, axis=1)
beta = 1
f_score = (1 + beta**2) * prec * recall / (beta**2 * prec + recall)

plt.figure()
sb.heatmap(con_mat, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Prediction')
plt.ylabel('Ground Truth')
plt.savefig('confusion_mat.png')

plt.figure()
sb.heatmap(prec.reshape((num_classes, 1)).T, annot=True)
plt.title('Precision')
plt.xlabel('Class')
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False,
    labelleft=False)
plt.savefig('precision.png')

plt.figure()
sb.heatmap(recall.reshape((num_classes, 1)), annot=True)
plt.title('Recall')
plt.ylabel('Classe')
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,      
    top=False,         
    labelbottom=False) 
plt.savefig('recall.png')

plt.figure()
sb.heatmap(f_score.reshape((num_classes, 1)).T, annot=True, square=True)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False,
    labelleft=False)
plt.title('F-Score, beta = {}'.format(beta))
plt.xlabel('Class')
plt.savefig('f_score.png')

#%% Visualize Model

import keract
from PIL import Image

idx = np.random.randint(0, n_test)
u = x_test[idx, :, :, :].reshape(1, img_rows, img_cols, 3)
activations = keract.get_activations(model, u, auto_compile=True)

# test image plotted
plt.figure()
plt.imshow(Image.fromarray(u[0, :, :, :], 'RGB'))
plt.axis('off')
pred_class = np.argmax(model.predict(u, verbose=1))
plt.title('Predicted #: {}'.format(pred_class))
plt.savefig('00_test_image.png')

#%% feature maps
keract.display_activations(activations, 
                            cmap='viridis', 
                            save=True, 
                            directory='.', 
                            data_format='channels_last')
# keract.display_heatmaps(activations, u, save=True)


#%% visualizing filters

layer_dict = dict([(layer.name, layer) for layer in model.layers])

layer_name = 'input'

# Grab the filters and biases for that layer
filters, biases = layer_dict[layer_name].get_weights()

# Normalize filter values to a range of 0 to 1 so we can visualize them
f_min, f_max = np.amin(filters), np.amax(filters)
filters = (filters - f_min) / (f_max - f_min)

# Plot first few filters
n_filters, index = 6, 1
for i in range(n_filters):
    f = filters[:, :, :, i]
    
    # Plot each channel separately
    for j in range(3):

        ax = plt.subplot(n_filters, 3, index)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.imshow(f[:, :, j], cmap='viridis')
        index += 1
plt.savefig('conv_1_filters.png')