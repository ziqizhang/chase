# MLP for Pima Indians Dataset with grid search via sklearn
#import tensorflow as tf
print("ok")
#single thread
##session_conf = tf.ConfigProto(
#  intra_op_parallelism_threads=2,
#  inter_op_parallelism_threads=5)
import os
#os.environ['THEANO_FLAGS']='floatX=32,device=gpu0'
os.environ['THEANO_FLAGS']="device=cpu,openmp=True"
#os.environ['THEANO_FLAGS']="openmp=True"

#tf.set_random_seed(2)
#session_conf = tf.ConfigProto(
 # intra_op_parallelism_threads=1,
 # inter_op_parallelism_threads=1)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

#sess = tf.Session(config=session_conf)
#with sess.as_default():
#  print(tf.constant(42).eval())

#sess = tf.Session(graph=tf.get_default_graph())
#K.set_session(sess)
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
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# grid search epochs, batch size and optimizer
optimizers = ['adam']
init = ['uniform']
epochs = [10,20,30,40,50]
batches = [5,10,15,20,25,30,35,40]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

print(datetime.datetime.now())
#K.clear_session()


from theano import function, config, shared, tensor
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
