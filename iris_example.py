"""@author: Tanzim"""
# Multiclass Classification with the Iris Flowers Dataset
import pandas as pd
# load dataset
dataset = pd.read_csv("iris.csv", header=None).values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]
# encode class values as integers
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
labelencoder_Y.fit_transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(labelencoder_Y)
# define baseline model
def baseline_model():
	# create model
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
from keras.wrappers.scikit_learn import KerasClassifier
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
from sklearn.model_selection import cross_val_score
results = cross_val_score(estimator, X, onehotencoder, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))