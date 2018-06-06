"""@author: Tanzim"""
# Baseline Model on the Sonar Dataset
import pandas  as pd
# load dataset
dataset= pd.read_csv("sonar.csv", header=None).values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
# encode class values as integers
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelencoder.fit_transform(Y)
# baseline
def create_baseline():
	# create model
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
    from keras.optimizers import SGD
    sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
estimators = []
from sklearn.preprocessing import StandardScaler
estimators.append(('standardize', StandardScaler()))
from keras.wrappers.scikit_learn import KerasClassifier
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=300, batch_size=16, verbose=0)))
from sklearn.pipeline import Pipeline
pipeline = Pipeline(estimators)
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
from sklearn.model_selection import cross_val_score
results = cross_val_score(pipeline, X, labelencoder, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))