import tensorflow as tf
import pandas as pd
import numpy as np

#from scipy.misc import  imsave
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers import Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dense, Flatten, Dropout
#from keras.optimizers import Adam

##################################################
#Preprocess the data
def flipped(image, measurement):
	image_flipped = np.fliplr(image)
	measurement_flipped = -measurement
	return image_flipped, measurement_flipped

##################################################
# Load and process driving log data.
driving_data = pd.read_csv('./data_own/driving_log.csv', sep = ',', header = None)

Center_image = []
Steering_angle = []
Image_train = []

                         

for y in Center_image:
    Image = mpimg.imread(y) #y==>Path
    Image_train.append(Image)

X_train = np.array(Image_train)#Center image
y_train = np.array(Steering_angle)#Steering angle

X_train, y_train = shuffle(X_train, y_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1)

##################################################
model = Sequential()

# set up lambda layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160, 320, 3)))

# Cropping Images
model.add(Cropping2D(cropping = ((70, 25), (0, 0)), input_shape = (160, 320, 3)))

# Convolutional Layer 1
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Convolutional Layer 2
model.add(Convolution2D(32, 3, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Convolutional Layer 3
model.add(Convolution2D(16, 3, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten
model.add(Flatten())

# Fully Connected Layer 1 and Dropout
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.5))

# Fully Connected Layer 2 and Dropout
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.5))

# Fully Connected Layer 3 and Dropout
model.add(Dense(20, activation = 'relu'))
model.add(Dropout(0.5))

# Final fully Connected Layer -- Sterring angle
model.add(Dense(1))

##################################################
# Compiling and training the model
model.compile(metrics = ['mean_squared_error'], loss = 'mean_squared_error', optimizer = 'Adam')
history_object = model.fit(X_train, y_train, nb_epoch = 5, batch_size = 100, verbose = 2, validation_data = (X_val, y_val))

### print the keys contained in the history object
#print(history_object.history.keys())

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
##################################################
# Saving model

#model.save_weights('model.h5')

#with open("model.json", "w") as json_file:
#    json_file.write(model.to_json())

model.save('model.h5')
print("Model Saved.")