"""@author: Tanzim"""
# Regression Example With Boston Dataset: Standardized and Larger
import pandas as pd
# load dataset
dataset = pd.read_csv("housing.csv", delim_whitespace=True, header=None).values

# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]
# define the model
def larger_model():
    from keras.models import Sequential
    from keras.layers import Dense
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# evaluate model with standardized dataset
estimators = []
from sklearn.preprocessing import StandardScaler
estimators.append(('standardize', StandardScaler()))
from keras.wrappers.scikit_learn import KerasRegressor
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
from sklearn.pipeline import Pipeline
pipeline = Pipeline(estimators)
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, random_state=0)
from sklearn.model_selection import cross_val_score
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))