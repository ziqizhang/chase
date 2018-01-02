# MLP for Pima Indians Dataset with grid search via sklearn
#import tensorflow as tf
from sklearn.cross_validation import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score

import os
os.environ['THEANO_FLAGS']="device=cpu,openmp=True"
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy

# Function to create model, required for KerasClassifier
def create_model(optimizer='rmsprop', init='glorot_uniform'):
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu'))
        model.add(Dense(8, kernel_initializer=init, activation='relu'))
        model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
print(datetime.datetime.now())
# fix random seed for reproducibility
seed = 1
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("/home/zqz/Work/data/pima-indians-diabetes.data", delimiter=",")
# split into input (X) and output (Y) variables


print(">>>>> n-fold")
X = dataset[:,0:8]
Y = dataset[:,8]

model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=5,verbose=0)
results = cross_val_score(model, X, Y, cv=5)
print(results.mean())


print(">>>>> grid search")
X_train_data, X_test_data, y_train, y_test = \
         train_test_split(dataset[:,0:8], dataset[:,8],
                          test_size=0.25,
                          random_state=42)

# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# grid search epochs, batch size and optimizer
optimizers = ['adam']
init = ['uniform']
epochs = [10]
batches = [5]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_result = grid.fit(X_train_data, y_train)

print("\tcrossfold running...{}".format(datetime.datetime.now()))
#nfold_predictions = cross_val_predict(grid.best_estimator_, X_train_data, y_train, cv=5)
print(cross_val_score(grid.best_estimator_, X_train_data, y_train, cv=5).mean())
#0.69827588  0.63478262  0.61739132  0.68695653  0.62608697
best_param_ann = grid.best_params_
print("\tbest params are:{}".format(best_param_ann))
best_estimator = grid.best_estimator_
heldout_predictions = best_estimator.predict(X_test_data)
print("\ttesting on the heldout...")
print(accuracy_score(y_test,heldout_predictions))
print(datetime.datetime.now())
#K.clear_session()



# from theano import function, config, shared, tensor
# import numpy
# import time
#
# vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
# iters = 1000
#
# rng = numpy.random.RandomState(22)
# x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
# f = function([], tensor.exp(x))
# print(f.maker.fgraph.toposort())
# t0 = time.time()
# for i in range(iters):
#     r = f()
# t1 = time.time()
# print("Looping %d times took %f seconds" % (iters, t1 - t0))
# print("Result is %s" % (r,))
# if numpy.any([isinstance(x.op, tensor.Elemwise) and
#               ('Gpu' not in type(x.op).__name__)
#               for x in f.maker.fgraph.toposort()]):
#     print('Used the cpu')
# else:
#     print('Used the gpu')
