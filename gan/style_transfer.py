from keras import backend as kb
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

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

textureImage = load_img(path=textureInPath, target_size=targetSize)
textureArray = img_to_array(textureImage)
textureArray = kb.variable(preprocess_input(np.expand_dims(textureArray, axis=0)), dtype='float32')

print(textureArray.shape)

seedImage = np.random.randint(256, size=(targetWidth, targetHeight, 3)).astype('float64')
seedImage = preprocess_input(np.expand_dims(seedImage, axis=0))
seedPlaceholder = kb.placeholder(shape=(1, targetWidth, targetHeight, 3))

def get_feature_reps(x, layer_names, model):
	featMatrices = []
	for ln in layer_names:
		selectedLayer = model.get_layer(ln)
		featRaw = selectedLayer.output
		featRawShape = kb.shape(featRaw).eval(session=tf_session)
		N_l = featRawShape[-1]
		M_l = featRawShape[1] * featRawShape[2]

		featMatrix = kb.reshape(featRaw, (M_l, N_l))
		featMatrix = kb.transpose(featMatrix)
		featMatrices.append(featMatrix)
	return featMatrices

def get_content_loss(F, P):
	cLoss = 0.5 * kb.sum(kb.square(F - P))
	return cLoss

def get_Gram_matrix(F):
	G = kb.dot(F, kb.transpose(F))
	return G

def get_style_loss(ws, Gs, As):
	sLoss = kb.variable(0.0)
	for w, G, A in zip(ws, Gs, As):
		M_l = kb.int_shape(G)[1]
		N_l = kb.int_shape(G)[0]
		G_gram = get_Gram_matrix(G)
		A_gram = get_Gram_matrix(A)
		sLoss = sLoss + (w * 0.25 * kb.sum(kb.square(G_gram - A_gram)) / (N_l**2 * M_l**2))
	return sLoss

def get_total_loss(seedPlaceholder, alpha=1.0, beta=10000.0):
	F = get_feature_reps(seedPlaceholder, layer_names=[cLayerName], model=gModel)[0]
	Gs = get_feature_reps(seedPlaceholder, layer_names=sLayerNames, model=gModel)
	contentLoss = get_content_loss(F, P)
	sytleLoss = get_style_loss(ws, Gs, As)
	totalLoss = alpha * contentLoss + beta * sytleLoss
	return totalLoss

def calculate_loss(seedImage):
	if seedImage.shape != (1, targetWidth, targetWidth, 3):
		seedImage = seedImage.reshape((1, targetWidth, targetHeight, 3))
	loss_fcn = kb.function([gModel.input], [get_total_loss(gModel.input)])
	return loss_fcn([seedImage])[0].astype('float64')

def get_grad(seedImage):
	if seedImage.shape != (1, targetWidth, targetHeight, 3):
		seedImage = seedImage.reshape((1, targetWidth, targetHeight, 3))
	grad_fcn = kb.function([gModel.input], 
						  kb.gradients(get_total_loss(gModel.input), [gModel.input]))
	grad = grad_fcn([seedImage])[0].flatten().astype('float64')
	return grad

from keras.applications import VGG16
from scipy.optimize import fmin_l_bfgs_b

tf_session = kb.get_session()
cModel = VGG16(include_top=False, weights='imagenet', input_tensor=inArray)
sModel = VGG16(include_top=False, weights='imagenet', input_tensor=textureArray)
gModel = VGG16(include_top=False, weights='imagenet', input_tensor=seedPlaceholder)
cLayerName = 'block4_conv2'
sLayerNames = [
				'block1_conv1',
				'block2_conv1',
				'block3_conv1',
				'block4_conv1',
				]

P = get_feature_reps(x=inArray, layer_names=[cLayerName], model=cModel)[0]
As = get_feature_reps(x=textureArray, layer_names=sLayerNames, model=sModel)
ws = np.ones(len(sLayerNames))/float(len(sLayerNames))

iterations = 10
x_val = seedImage.flatten()
xopt, f_val, info= fmin_l_bfgs_b(calculate_loss, x_val, fprime=get_grad,
							maxiter=iterations, disp=True)

