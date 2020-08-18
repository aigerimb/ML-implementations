import numpy as np
import pandas as pd 
from scipy.special import expit, logit
from sklearn.utils.extmath import softmax
import matplotlib.pyplot as plt


# this class creates individual layers of neural network 
class Layers(object):
    #n_input - dimenstions of the input 
    #n_neurons - # of neurons 
    #activation - specifies activation function e.g. softmax, sigmoid 
    def __init__(self, n_input, n_neurons, activation):
        self.w = np.random.randn(n_input, n_neurons)
        self.b = np.random.randn(n_neurons)
        self.activation = activation 
        self.error = 0
        self.delta = 0
        
    # returns output of sigmoid function     
    def sigmoid(self, Z):

        return expit(Z)
    
    # takes derivative of sigmoid function 
    def derivative_sigmoid(self, s):
    
        return s*(1-s)
    
    # adds activation function for a layer 
    def activate(self, X):
      
        if self.activation == 'sigmoid': 
            Z = np.dot(X, self.w) 
            self.last_activation = self.sigmoid(Z) 
    
        if self.activation == 'softmax': 
            Z = np.dot(X, self.w) 
            self.last_activation = softmax(Z)
    
        return self.last_activation


 
# this class performs forward and back propogation through neural network 
class NeuralNetwork(object):
  # layer params should be: n_input, n_neurons format 
  # lr - learning rate to perform stochastic gradient descent 
    def __init__(self, lr, rp):
        self.layers = []
        self.lr = lr
        self.rp = rp 
        self.layers = []
    
    # creates each layer of neural network 
    def add_layer(self, l):

      self.layers.append(l)

    # performs forward propogation of inputs 
    def feedforward(self, X):
        inputs = X
        for i in range(len(self.layers)):
            l = self.layers[i]
            output = l.activate(inputs)
            inputs = output
        return output 
    
    # predict class 
    def predict(self, X):
        ff = self.feedforward(X)
        
        return np.argmax(ff, axis=1)
    
    # compute cross-entropy using residual error 
    def cross_entropy(self, pred, real):
        n_samples = real.shape[0]
        res = pred - real
        return res/n_samples
    
    # performs backpropogation through neural network 
    def backpropagation(self, X, Y):
        
        output = self.feedforward(X)

        # calculate errors for each layer 
        for i in reversed(range(len(self.layers))):
      
            layer = self.layers[i]

            if i == len(self.layers)-1:
                layer.error = self.cross_entropy(Y, output) 
                layer.delta = layer.error
      
            else:
      
                next_layer = self.layers[i+1]
                layer.error = np.dot(next_layer.delta, next_layer.w.T)
                layer.delta = layer.error*layer.derivative_sigmoid(layer.last_activation)

       
        # update weights using calculated errors, inputs and learning rate 
        for i in range(len(self.layers)):

            layer = self.layers[i]
      
            if i == 0:
                inputs = np.atleast_2d(X) 
            else:
                last_layer = self.layers[i-1]
                inputs = np.atleast_2d(last_layer.last_activation)
            layer.w += np.dot(inputs.T, layer.delta)*self.lr

    # trains neural network by backpropogation for max number of epochs
    def train(self, X, Y, max_epochs):
        mses = []
        for i in range(max_epochs):
            self.backpropagation(X, Y)
    
    # tests predictions of neural network and computes accuracy 
    def test(self, X_test, Y_test):
        y_predict = nn.predict(X_test)
        c =0
        for i in range(len(y_predict)):
            if y_predict[i] ==np.argmax(Y_test[i]):
                c+=1
        return c/len(y_predict)



# normalizes features 
def normalization(X):
    X_norm = np.zeros(X.shape)
    for j in range(X.shape[1]):
        max_v = max(X[:, j])
        min_v = min(X[:, j])
        if max_v != min_v:
            for i in range(X.shape[0]):
                X_norm[i, j] = (X[i, j]- min_v)/(max_v-min_v)
        else:
            X_norm[:, j] = 1
    return X_norm 

# returns onehot encoding of labels 
def onehot(Y):
    Y_ones = np.zeros([len(Y), n_cl])
    for i in range(len(Y)):
        Y_onehot = np.zeros(n_cl)
        Y_onehot[Y[i]-1] = 1
        Y_ones[i, :] = Y_onehot
    return Y_ones 


# splits given data into training and testing sets 
def split_data(X_norm,  Y_ones, train_size):
    n = X_norm.shape[0]
    X_train = X_norm[:train_size, :]
    X_test = X_norm[train_size:int(n+1), :]
    Y_train = Y_ones[:train_size, :]
    Y_test = Y_ones[train_size:int(n+1), :]
    
    return X_train, X_test, Y_train, Y_test 




          
      

# reading data and preparation
data = pd.read_csv("Image_Segmentation.csv")
# training data size 
train_size = 2000
# total number of classes 
n_cl = 7
X = data.iloc[:, :-1].values
Y =  data.iloc[:, -1].values
Y_ones = onehot(Y)
X_norm = normalization(X)
X_train, X_test, Y_train, Y_test  = split_data(X_norm, Y_ones, train_size)


# start training 
n_features = 19
n_neurons = 5
n_output = 7


results = []
iters = 10
for n in range(iters):
    nn = NeuralNetwork(0.1, 0.1)
    nn.add_layer(Layers(n_features, n_neurons, 'sigmoid'))
    nn.add_layer(Layers(n_neurons, n_output, 'softmax'))

    nn.train(X_train, Y_train, 10000)

    accuracy = nn.test(X_test, Y_test)
    print("accuracy", accuracy)
    results.append(accuracy)
        
