from keras import backend as kb
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

inPath = "/Users/angel/Desktop/git/coding-and-cats/imgs/backgrounds/fry-code.jpeg"
textureInPath = "texture.jpg"
outPath = "img_out/ .jpg"

targetHeight = 512
targetWidth = 512
targetSize = (targetHeight, targetWidth)

inImage = load_img(path=inPath, target_size=targetSize)
inArray = img_to_array(inImage)
inArray = kb.variable(preprocess_input(np.expand_dims(inArray, axis=0)), dtype='float32')

print(inArray.shape)

textureImage = load_img(path=textureInPath, target_size=target_size)
textureArray = img_to_array(textureImage)
textureArray = kb.variable(preprocess_input(np.expand_dims(textureArray, axis=0)), dtype='float32')

print(textureArray.shape)