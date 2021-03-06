import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13] #matrix features independent variable
y = dataset.iloc[:, 13] #label depedent variable

# Ecoding Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer

#Encode Categorical and Scale Features
preprocess = make_column_transformer(
	(StandardScaler(), ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']),
	(OneHotEncoder(), ['Geography', 'Gender'])
	)

X = pd.DataFrame(preprocess.fit_transform(X)).drop([8, 11], axis=1) #drop to avoid dummy variable trap -- dropping 1 geo, and 1 gender
X = X.values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Making the ANN
import keras
# from keras.models import Sequential #for initalizing the neural net
# from keras.layers import Dense #for making layers
# from keras.layers import  Dropout #Dropout randomly disables neurons -- used to prevent overfitting, or correct it. Overfittign occurs when correlations become to tight
#by using dropout we remove the possibility of this occuring since different neurons are used at each pass

# #For a problem like churn we're dealing with a classification problem
# #We first initialize the classifier
# classifier = Sequential()

# #adding layers to the ANN -- inital layer, and first hidden layer
# classifier.add(
# 	Dense(
# 		output_dim = 6, #number of hidden layer nodes
# 		init= 'uniform', #how the weights are distributed on the inital nodes 
# 		activation='relu', #activation function
# 		input_dim=11 #number of nodes to start with, just match the columns in your feature data
# 		)
# 	)
# classifier.add(Dropout(p=0.1)) #start low, don't go over 4.5 or you right underfitting
# #adding the second hidden layer
# classifier.add(
# 	Dense(
# 		output_dim = 6, #number of hidden layer nodes
# 		init= 'uniform', #how the weights are distributed on the inital nodes 
# 		activation='relu', #activation function
# 		)
# 	)
# classifier.add(Dropout(p=0.1))

# #adding the output layer
# classifier.add(
# 	Dense(
# 		output_dim=1, #there is only one outcome when binary
# 		init='uniform',
# 		activation='sigmoid' #chose signmoid because likelyhood of churn is on a specturm -- lets us rank, also either 1, or 0, so good choice
# 		)
# 	)

# #Compiling the ANN
# classifier.compile(
# 	optimizer='adam', #a very efficient sochastic gradient descent algorithm
# 	loss='binary_crossentropy', #categorical_crossentrophy if more non-binary -- choices can correpsond to the final activation function. this is what model is trying to minimize via Gradient Descent
# 	metrics=['accuracy'] #list of metrics to evaluate 
# 	)

# #Fit the model to the training set
# classifier.fit(
# 	X_train, 
# 	y_train, 
# 	batch_size=10,
# 	nb_epoch=100
# 	)

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
# y_pred = (y_pred > 0.5)

# exit()
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# #accuracy
# print((cm[0][0] + cm[1][1]) / 2000)

# ##Now we Evaluate the ANN using k-fold cross validation with k=10
# #The idea is that we preform multiple rounds of training and evaluate the results
# #Average Accuracy of the 10 rounds is used to validate the model
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score
# from keras.models import Sequential
# from keras.layers import Dense

# #wrap the classifier into a function
# def build_classifier():
# 	classifier = Sequential()
# 	classifier.add(Dense(
# 		units=6,
# 		kernel_initializer='uniform',
# 		activation='relu',
# 		input_dim=11
# 		)
# 	)
# 	classifier.add(Dense(
# 		units=6,
# 		kernel_initializer='uniform',
# 		activation='relu'
# 		)
# 	)
# 	classifier.add(Dense(
# 		units=6,
# 		kernel_initializer='uniform',
# 		activation='sigmoid'
# 		)
# 	)
# 	classifier.add(
# 	Dense(
# 		units=1, #there is only one outcome when binary
# 		kernel_initializer='uniform',
# 		activation='sigmoid' #chose signmoid because likelyhood of churn is on a specturm -- lets us rank, also either 1, or 0, so good choice
# 		)
# 	)
# 	classifier.compile(
# 		optimizer='adam',
# 		loss='binary_crossentropy',
# 		metrics=['accuracy']
# 		)
# 	return classifier

# #create the classifier that will cross fold
# #this part is equivalent to .fit()
# classifier = KerasClassifier(
# 	build_fn=build_classifier,
# 	batch_size=10,
# 	epochs=100
# 	)

# #runs the validation
# accuracies = cross_val_score(
# 	estimator=classifier,
# 	X=X_train, #train set
# 	y=y_train, #labels
# 	cv=10,  #number of times to run the model
# 	n_jobs=-1 #how man cpu's to use
# 	)

# mean = accuracies.mean()
# variance = accuracies.std()
# print(mean)
# print(variance)
# print('DONE')
# exit()

#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense


def build_classifier(optimizer, num_layer_1, num_layer_2):
	classifier = Sequential()
	classifier.add(Dense(
		units=num_layer_1,
		kernel_initializer='uniform',
		activation='relu',
		input_dim=11
		)
	)
	classifier.add(Dense(
		units=num_layer_2,
		kernel_initializer='uniform',
		activation='relu'
		)
	)
	# classifier.add(Dense(
	# 	units=num_layer_2,
	# 	kernel_initializer='uniform',
	# 	activation='sigmoid'
	# 	)
	# )
	classifier.add(
	Dense(
		units=1, #there is only one outcome when binary
		kernel_initializer='uniform',
		activation='sigmoid' #chose signmoid because likelyhood of churn is on a specturm -- lets us rank, also either 1, or 0, so good choice
		)
	)
	classifier.compile(
		optimizer='adam',
		loss='binary_crossentropy',
		metrics=['accuracy']
		)
	return classifier

#this takes over a day to run
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {
	'batch_size': [15, 25],
	'epochs': [500, 700],
	'optimizer': ['adam', 'rmsprop'],
	'num_layer_1': [4, 6, 8, 11],
	'num_layer_2': [4, 6, 8, 11]
}

grid_search = GridSearchCV(
	estimator=classifier,
	param_grid=parameters,
	scoring='accuracy',
	cv=10)

grid_search = grid_search.fit(X_train, y_train)
#results
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

