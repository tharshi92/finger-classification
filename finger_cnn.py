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

batch_size = 16
num_classes = 6
epochs = 100
split = 0.25
pat = 10

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
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

#%% Visualize Model
import matplotlib.pyplot as plt
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
#keract.display_heatmaps(activations, u, save=True)

#%% Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sb

y_pred = model.predict(x_test, verbose=1)
con_mat = confusion_matrix(y_test.argmax(axis=1), 
                          y_pred.argmax(axis=1))

plt.figure()
sb.heatmap(con_mat, annot=True)
plt.title('Confusion Matrix')
plt.savefig('confusion_mat.png')