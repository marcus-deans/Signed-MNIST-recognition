import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras import layers
from keras.layers import Input, Dense, Activation, MaxPool2D, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Import the American Sign Language MNIST Dataset hosted on Kaggle
xtrain = pd.read_csv("../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")
xtest  = pd.read_csv("../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")

#The MNIST Datset is already labeled - extract and save
ytrain = xtrain['label']
ytest  = xtest['label']

#Remove labels from xtrain and xtest
del xtrain['label']
del xtest['label']

#Now normalize and reshape the data, to fit the CNN and facilitate faster learning
xtrain = (xtrain/255).values.reshape(-1, 28, 28 ,1)
xtest  = (xtest/255).values.reshape(-1, 28, 28, 1)

#Perform data augmentation which will artifically grow dataset, allowing CNN to better generalize learning
#Images were rotated, zoomed, and shifted by up to 15% of the relevant factor (degrees, zoom, width, height respectively)
#Note that neither horizontal nor vertical flips were employed, due to the specifics of American Sign Language.
augdata = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.15, # Randomly zoom image
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

#Apply data augmentation to the training set
augdata.fit(xtrain)

#Reducing learning rate of CNN during plateaus to continue progress
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

#Generate CNN Model
model = Sequential()

#First CONV Layer with 16 3x3 Kernels with stride of 1, then 2x2 pooling
model.add(Conv2D(16 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

#Second CONV Layer with 32 3x3 Kernels with stride of 1, then Dropout regularization and 2x2 pooling
model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2)) #Dropout for regularization
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

#Third CONV Layer with 64 3x3 Kernels with stride of 1, then Dropout regularization and 2x2 pooling
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2)) #Dropout for regularization
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

#Fourth CONV Layer with 128 3x3 Kernels with stride of 1, then Dropout regularization and 2x2 pooling
model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

#Flatten the 3-dimensional tensor into a vector to input into fully-connected layers
model.add(Flatten())

#First Fully-Connected layer with 256 nodes and Dropout regularization
model.add(Dense(units = 256 , activation = 'relu'))
model.add(Dropout(0.3))

#Second Fully-Connected layer with 128 nodes and Dropout regularization
model.add(Dense(units = 128 , activation = 'relu'))
model.add(Dropout(0.3))

#Final output layer with softmax function to categorize as one of the 24 classifications from Signed-MNIST set.
model.add(Dense(units = 25 , activation = 'softmax'))

#Compile model using standard adam optimizer. Note that sparse_categorical_crossentropy is used as these are integers, not one-hot vectors.
model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

#Create developmental set for usage in training by subdividing training set
xtrain, xdev, ytrain, ydev = train_test_split(xtrain, ytrain, test_size = 0.2, random_state = 42)

#Fit CNN to training set and perform validation with the dev set
history = model.fit(augdata.flow(xtrain,ytrain, batch_size = 128) ,epochs = 25 , validation_data = (xdev, ydev) ,
                    callbacks = [learning_rate_reduction])

#Evaluate CNN accuracy on the final test set
print("Test Accuracy " , model.evaluate(xtest,ytest)[1]*100 , "%")