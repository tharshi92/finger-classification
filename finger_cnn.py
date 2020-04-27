# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 20:22:43 2020

@author: tharshi
"""

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ELU
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
import os

from sklearn.metrics import confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt

# load data
X = np.load('../finger-counter-data/images.npy')

# input image dimensions
n_hands, img_rows, img_cols = X.shape[0], X.shape[1], X.shape[2]

# load labels
Y = np.load('../finger-counter-data/labels.npy')

# define input shape for model
input_shape = (img_rows, img_cols, 3)
#%%
if os.path.isfile('indicies.npy'):
    print('Data has already been partitioned, loading...', end='')
    indices = np.load('indicies.npy')
    print('Done.\n')
else:
    indices = np.arange(n_hands)
    np.random.shuffle(indices)
    
    np.save('indicies.npy', indices)

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


#%% Mode

batch_size = 128
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

earlystop = EarlyStopping(verbose=True,
                          patience=pat,
                          monitor='val_loss')
if os.path.isfile('cnn_weights.h5'):
    print('Weights already exist, loading...', end='')
    model.load_weights('cnn_weights.h5')
    print('Done.\n')
else:
    model.summary()
    
    print('Training model...\n')
    hist = model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1, 
              shuffle=True, 
              validation_split=split,
              callbacks=[earlystop])
    model.save_weights('cnn_weights.h5')
    print('\nDone.\n')
    
    print('Plotting training history...', end='')

    # plot acc and loss
    fig4 = plt.figure()
    ax4_1 = plt.subplot(211)
    ax4_2 = plt.subplot(212, sharex=ax4_1)
    
    ax4_1.semilogy(hist.history['accuracy'], label='train')
    ax4_1.semilogy(hist.history['val_accuracy'], label='validation')
    
    ax4_2.semilogy(hist.history['loss'], label='train')
    ax4_2.semilogy(hist.history['val_loss'], label='validation')
    
    ax4_1.set_title('Training History')
    plt.xlabel("epochs")
    ax4_1.set_ylabel('accuracy')
    ax4_2.set_ylabel('loss')
    plt.setp(ax4_1.get_xticklabels(), visible=False)
    
    ax4_1.legend(loc='lower right')
    ax4_2.legend(loc='lower left')
    plt.savefig("training_history.png", dpi=300)
    
    print('Done.\n')

print('Calculating test set stats...', end='')
score = model.evaluate(x_test, y_test, verbose=0)
print('Done.')
print('Test loss:', score[0])
print('Test accuracy:', score[1], '\n')


#%% Confusion matrix


y_pred = model.predict(x_test, verbose=1)
con_mat = confusion_matrix(y_test.argmax(axis=1), 
                          y_pred.argmax(axis=1))

prec = np.diag(con_mat) / np.sum(con_mat, axis=0)
recall = np.diag(con_mat) / np.sum(con_mat, axis=1)

beta = 1
f_score = (1 + beta**2) * prec * recall / (beta**2 * prec + recall)

print('Plotting and saving metric figures...', end='')

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

print('Done.\n')

#%% Visualize Model

import keract
from PIL import Image

idx = np.random.randint(0, n_test)
u = x_test[idx, :, :, :].reshape(1, img_rows, img_cols, 3)
activations = keract.get_activations(model, u, auto_compile=True)

# test image plotted
print('Plotting and saving test image...', end='')

plt.figure()
plt.imshow(Image.fromarray(u[0, :, :, :], 'RGB'))
plt.axis('off')
pred_class = np.argmax(model.predict(u, verbose=1))
plt.title('Predicted #: {}'.format(pred_class))
plt.savefig('./visuals/00_test_image.png')
print('Done.\n')

#%% feature maps

print('Plotting and saving activations...\n')

keract.display_activations(activations, 
                            cmap='viridis', 
                            save=True, 
                            directory='./visuals/', 
                            data_format='channels_last')


# keract.display_heatmaps(activations, u, save=True)

print('\nDone.\n')


#%% Layer Viz - Filters
#Select a convolutional layer
layer = model.layers[0]

#Get weights
kernels, biases = layer.get_weights()

#Normalize kernels into [0, 1] range for proper visualization
kernels = np.moveaxis(kernels, 3, 0)
spread = (np.max(kernels, axis=0) - np.min(kernels, axis=0))
kernels = (kernels - np.min(kernels, axis=0)) / spread
kernels = np.moveaxis(kernels, 0, -1)

#Weights are usually (width, height, channels, num_filters)
#Save weight images

# Plot filters
n_filters, index = 8, 1
fig = plt.figure()
for i in range(n_filters):
    f = kernels[:, :, :, i]

    ax = plt.subplot(2, n_filters // 2, index)
    plt.title('filter #{}'.format(index))
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.imshow(f)
    index += 1
plt.suptitle('Filters for layer: {}'.format(layer_name))
plt.savefig('./visuals/conv_1_filters.png')