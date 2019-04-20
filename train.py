import theano.tensor as T
import theano
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# get the data
train_df = pd.read_csv('digit-recognizer/train.csv')
X = np.array(train_df.iloc[:,1:].astype(np.float32))/255
Y = np.array(train_df.iloc[:,0].astype(np.float32)).reshape((X.shape[0],1))
# X = np.random.randn(1000,2)
# X[500:,:] += [3,3]
# Y = np.zeros((1000,1))
# Y[500:,0] = 1

# create an indicator matrix for Ytrain
Yind = np.zeros((Y.shape[0],10))
for i in range(len(Yind)):
    Yind[i,int(Y[i,0])] = 1
Y = np.copy(Yind).astype(np.float32)

# split into training and test sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,random_state=24,test_size=0.3)

# delete unwanted variables
del Yind, X,Y

# plt.imshow(Xtrain[1751,:].reshape((28,28)))
# plt.show()
# print(np.argmax(Ytrain[1751,:]))


# set hyperparameters
N = Xtrain.shape[0]
D = Xtrain.shape[1]
M1 = 300
M2 = 200
K = Ytrain.shape[1]
learning_rate = 0.0001
decay = 0.99
iter = 72
every = 2
bs = 1000

# initialize weights
w1_cache_init = np.ones((D,M1))
b1_cache_init = np.ones((M1))
w2_cache_init = np.ones((M1,M2))
b2_cache_init = np.ones((M2))
w3_cache_init = np.ones((M2,K))
b3_cache_init = np.ones((K))
w1_init = np.random.randn(D,M1) / np.sqrt(D)
b1_init = np.random.randn(M1)
w2_init = np.random.randn(M1,M2) / np.sqrt(M1)
b2_init = np.random.randn(M2)
w3_init = np.random.randn(M2,K) / np.sqrt(M2)
b3_init = np.random.randn(K)

# set theano shared variables and matrices
X = T.matrix(name='X')
Y = T.matrix(name='Y')
w1_cache = theano.shared(w1_cache_init, name='w1_cache')
b1_cache = theano.shared(b1_cache_init, name='b1_cache')
w2_cache = theano.shared(w2_cache_init, name='w2_cache')
b2_cache = theano.shared(b2_cache_init, name='b2_cache')
w3_cache = theano.shared(w3_cache_init, name='w3_cache')
b3_cache = theano.shared(b3_cache_init, name='b3_cache')
w1 = theano.shared(w1_init, name='w1')
b1 = theano.shared(b1_init, name='b1')
w2 = theano.shared(w2_init, name='w2')
b2 = theano.shared(b2_init, name='b2')
w3 = theano.shared(w3_init, name='w3')
b3 = theano.shared(b3_init, name='b3')

# define output
Yhat = T.nnet.softmax(T.dot(T.nnet.relu((T.dot(T.nnet.relu((T.dot(X,w1) + b1)),w2) + b2)),w3) + b3)

# define cost function
cost = -((Y * T.log(Yhat)).sum())

# perform updates
w1_cache_upd = (decay) * w1_cache + (1-decay) * T.grad(cost,w1)**2
b1_cache_upd = (decay) * b1_cache + (1-decay) * T.grad(cost,b1)**2 
w2_cache_upd = (decay) * w2_cache + (1-decay) * T.grad(cost,w2)**2 
b2_cache_upd = (decay) * b2_cache + (1-decay) * T.grad(cost,b2)**2 
w3_cache_upd = (decay) * w3_cache + (1-decay) * T.grad(cost,w3)**2 
b3_cache_upd = (decay) * b3_cache + (1-decay) * T.grad(cost,b3)**2 
w1_upd = w1 - learning_rate * ( T.grad(cost,w1) ) / T.sqrt(w1_cache + 10e-7)
b1_upd = b1 - learning_rate * ( T.grad(cost,b1) ) / T.sqrt(b1_cache + 10e-7)
w2_upd = w2 - learning_rate * ( T.grad(cost,w2) ) / T.sqrt(w2_cache + 10e-7)
b2_upd = b2 - learning_rate * ( T.grad(cost,b2) ) / T.sqrt(b2_cache + 10e-7)
w3_upd = w3 - learning_rate * ( T.grad(cost,w3) ) / T.sqrt(w3_cache + 10e-7)
b3_upd = b3 - learning_rate * ( T.grad(cost,b3) ) / T.sqrt(b3_cache + 10e-7)

# create theano functions
train = theano.function(inputs=[X,Y], outputs=[], updates=[(w1_cache,w1_cache_upd),(b1_cache,b1_cache_upd),(w2_cache,w2_cache_upd),(b2_cache,b2_cache_upd),(w3_cache,w3_cache_upd),(b3_cache,b3_cache_upd),(w1,w1_upd),(b1,b1_upd),(w2,w2_upd),(b2,b2_upd),(w3,w3_upd),(b3,b3_upd)])
get_prediction = theano.function(inputs=[X, Y], outputs=[cost, Yhat])


traincer = []
testcer = []
# batch SGD
for i in range(iter):
    Xtrain,Ytrain = shuffle(Xtrain,Ytrain)
    for j in range(len(Xtrain)//bs):
        Xbat = Xtrain[j*bs:(j+1)*bs,:]
        Ybat = Ytrain[j*bs:(j+1)*bs,:]

        train(Xbat,Ybat)
    if i%every == 0:
        traincst,trainyh = get_prediction(Xtrain,Ytrain)
        testcst, testyh = get_prediction(Xtest,Ytest)
        print('iteration: ' + str(i) + ' cost: ' + str(traincst) + ' classification rate: ' + str(np.mean(np.argmax(trainyh,axis=1)==np.argmax(Ytrain,axis=1))))
        traincer.append(traincst)
        testcer.append(testcst)

# simple SGD
# for i in range(iter):
#     train(Xtrain,Ytrain)
#     if i%every == 0:
#         traincst,trainyh = get_prediction(Xtrain,Ytrain)
#         testcst, testyh = get_prediction(Xtest,Ytest)
#         print('iteration: ' + str(i) + ' cost: ' + str(traincst) + ' classification rate: ' + str(np.mean(np.argmax(trainyh,axis=1)==np.argmax(Ytrain,axis=1))))
#         traincer.append(traincst)
#         testcer.append(testcst)

testcst, testyh = get_prediction(Xtest,Ytest)
print('Final test classification rate is: ' + str(np.mean(np.argmax(testyh,axis=1)==np.argmax(Ytest,axis=1))))
plt.plot(traincer,c='r',label='training cost')
plt.plot(testcer,c='b',label='test cost')
plt.legend()
plt.show()

pd.DataFrame(w1.eval()).to_csv('w1.csv')
pd.DataFrame(b1.eval()).to_csv('b1.csv')
pd.DataFrame(w2.eval()).to_csv('w2.csv')
pd.DataFrame(b2.eval()).to_csv('b2.csv')
pd.DataFrame(w3.eval()).to_csv('w3.csv')
pd.DataFrame(b3.eval()).to_csv('b3.csv')
