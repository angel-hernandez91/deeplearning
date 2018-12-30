# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize the CNN
classifier = Sequential()

# Start with the convolution layer
classifier.add(Convolution2D(
  32, #number of feature maps / filters, 32 is common practice to start with, scale up by powers of 2 in other layers
  (3,3), #rows in feature detector by columns in feature detector
  input_shape=(64, 64, 3), #size of the image -- all images must be in this format / size, since we have color images we set channels = 3. black/white uses 1
  #the order above is for tensor flow backend, theano reverses channels and size
  activation='relu' #use relu to remove negative pixels if they occur, this ensures we non-linearity
  ))

# Apply Max Pooling to reduce size of feature maps -- this improves performance and reduces complexity
classifier.add(MaxPooling2D(
  pool_size=(2, 2) # the rows and columns of the pooling matrix
  ))

# Adding a second Convolution Layer can Improve Accuracy
classifier.add(Convolution2D(
  32,
  (3,3),
  #input_shape=(64, 64, 3), this does not need to included here because we already know what the input shape is (it's the size of pooled feature maps)
  activation='relu'
  ))
classifier.add(MaxPooling2D(
  pool_size=(2, 2)
  ))

# Flatten the Max Pooling Layer
classifier.add(Flatten()) # contains all information about the spatial structure of the images -- this will feed into the first layer of the CNN

# First Fully Connected Layer
classifier.add(Dense(
  units=128, #common practice of using powers of 2 -- roughly half the size of the input from the Flattened Vector
  activation='relu'
  ))

#Output layer of the model
classifier.add(Dense(
  units=1, #only one output
  activation='sigmoid' #because we have binary output cat or dog
  ))

#Compile the CNN
classifier.compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics=['accuracy']
  )

#preprocess images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
  rescale=1.0/255, #scaled down to represent the 256 color scale
  shear_range=0.2, #default random transform
  zoom_range=0.2, #default random transform
  horizontal_flip=True #images are flipped
  )

test_datagen = ImageDataGenerator(rescale=1.0/255)

training_set = train_datagen.flow_from_directory(
  'dataset/training_set', #path to images
  target_size=(64, 64), #size of the images the CNN expected
  batch_size=32, #number of images that will go through the CNN before the weights are updated
  class_mode='binary' #since we have only two classes this is binary
  )

test_set = test_datagen.flow_from_directory(
  'dataset/test_set',
  target_size=(64,64),
  batch_size=32,
  class_mode='binary'
  )

classifier.fit_generator(
  training_set,
  steps_per_epoch=8000, #number of images in the training set
  epochs=25, #how many runs
  validation_data=test_set, #the data set we validate our model on
  validation_steps=2000, #number of images in the test set
  verbose=1,
  use_multiprocessing=True
  )

