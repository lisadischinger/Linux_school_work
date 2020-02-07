"""
Lisa Dischinger
2/9/20
Deep Learning Assignment 2
This will implament a one hidden layer fully connected Neural Network.
Backapropagation will use the cross-entropy loss function. This will also use ReLU as the hidden node activation
function but a sigmoidal activation at the final layer .

I will be testing out the capabilities of this neural network by playing with the batch size, learning rate, and with
the number of hidden units
"""

from __future__ import division
from __future__ import print_function

import sys
try:
   import _pickle as pickle
except:
   import pickle

import numpy as np
from math import log, exp

# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step


class LinearTransform(object):
    def __init__(self, number_of_inputs, number_of_outputs, n_b):
        # DEFINE __init function and create handles for the weights and biases
        # these will be the initial guesses
        self.i = number_of_inputs     # number of input values
        self.j = number_of_outputs     # number of outputs
        self.n_b = n_b              # batch size
        self.W = np.random.uniform(0, 1, size=(self.i, self.j))  # weights that go from the input into this layer
        self.B = np.random.uniform(0, 1, size=(self.j, 1))      # biases for the hidden layer linear transform
        self.d_W = np.zeros([self.i, self.j, self.n_b])         # temporary list of changes needed
        self.d_B = np.zeros([self.j, 1, self.n_b])
        self.S = np.zeros([self.j, 1])                          # output vector from linear transform

    def forward(self, x):
        # x is [ix1], w_LT is [ixj], b is [1xj], Z = [jx1]
        self.x = x                  # need to remember the input for back propagation
        for j in range(self.j):
            sum = 0
            for i in range(self.i):
                sum += (x[i] * self.W[i, j])
            self.S[j] = sum + self.B[j]
        return self.S

    def backward(self, grad_output, b, LR=0.0, M=0.0, l2_penalty=0.0):
        """ grad_output is the delta calulcated before accounting for this last step of linearization
            learning_rate is the amount we are allowing the gradient to correct our weights
            b is the batch number"""
        dE_dW = np.zeros((self.i, self.j))
        plane_of_gradients = np.zeros([self.i, self.j])
        for j in range(self.j):
            self.d_B[j, 0, b] = grad_output[j]
            for i in range(self.i):
                dE_dW[i, j] = grad_output[j] * self.x[i]
                plane_of_gradients[i, j] = -LR * dE_dW[i, j]

        try:
            self.G = np.append(self.G, plane_of_gradients,
                               axis=2)  # add this new column of gradients for momentum purposes
        except:
            self.G = np.atleast_3d(plane_of_gradients)              # creating of the 3D gradient list for momentum

        m_gradients = self.calucate_Momentum(M)
        for j in range(self.j):
            for i in range(self.i):
                self.d_W[i, j, b] = m_gradients[i, j]            # calculate the final gradient due to momentum
        return dE_dW

    def calucate_Momentum(self, M):
        # will apply momentum to the gradients to provide one final gradient that this round of weight updates will be
        # effected by. M is the momentum coefficient. This is used to better help us not get stuck in any saddle points
        n_row = self.G.shape[0]
        n_col = self.G.shape[1]                     # total number of gradient stored in list row
        n_depth = self.G.shape[2]
        gradients = np.zeros([n_row, n_col])
        for row in range(n_row):
            for col in range(n_col):
                i = 0
                for g in self.G[row, col, :]:
                    gradients[row, col] -= pow(M, n_col-i) * g             # M^(total - i) * g[i]
                    i += 1
        return gradients

    def update_weights(self):
        # call this to do the work of updating the weights and biases
        for j in range(self.j):
            dB = np.average(self.d_B[j, 0, :])
            self.B[j, 0] += dB
            for i in range(self.i):
                dW = np.average(self.d_W[i, j, :])
                self.W[i, j] += dW


class ReLU(object):                 # This is a class for a ReLU layer max(x,0)
    def __init__(self):
        dud = 47

    def forward(self, x):
        self.x = x                      # will need to reference this examples forward input to the function
        z = np.zeros(x.shape)
        for i in range(x.shape[0]):
            if x[i] > 0:
                z[i] = x[i]
            else:
                z[i] = 0
        return z

    def backward(self, dE_dZ1):
        dR_dS1 = np.zeros(self.x.shape)
        for i in range(self.x.shape[0]):    # diferentiating ReLU to x the input
            if self.x[i] > 0:
                dR_dS1[i] = 1 * dE_dZ1[i]
            else:
                dR_dS1[i] = 0
        return dR_dS1


class SigmoidCrossEntropy(object):
    # This is a class for a sigmoid layer followed by a cross entropy layer, the reason
    # this is put into a single layer is because it has a simple gradient form
    def __init__(self, num_output):
        self.k = num_output
        self.z_SCC = np.zeros(self.k)   # output from the sigmoidal activation function
        self.E_SCC = np.zeros(self.k)   # output from the cross entropy loss function

    def forward(self, s, y):
        """ s is the sum found through the linear transformation that occurs right before this step
            y is the target values for this example"""
        self.y = y
        for k in range(self.k):
            # sigmoidal activation function
            self.z_SCC[k] = 1 / (1 + exp(-s[k]))
            # cross entropy loss function
            self.E_SCC[k] = y[k]*log(self.z_SCC[k]) + (1-y[k])*log(1-self.z_SCC)

        return self.E_SCC

    def backward(self):
        # dE/dz = -y*/z + (1-y*)/(1-z)
        # dz/ds = z*(1-z)
        dE_ds = np.zeros(self.y.shape)
        for k in range(self.k):
            dE_ds[k] = self.z_SCC[k] - self.y[k]

        return dE_ds        # the delta based off of the loss function and sigmoid


class MLP(object):
    # This is a class for the Multilayer perceptron
    def __init__(self, input_dims, hidden_units, num_outputs, n_batch):
        self.i = input_dims                         # [i x 1]
        self.j = hidden_units                       # [jx1]
        self.k = num_outputs
        self.n_b = n_batch                          # number of time we calculate the deltas before averaging and actually applying

        # randomly initialize weights and biases
        # self.W1 = np.random.uniform(0, 1, size=(n_i, n_j))  # weights between input and hidden layer
        # self.W2 = np.random.uniform(0, 1, size=(n_j, n_k))  # weights between hidden layer and last layer
        # self.B1 = np.random.uniform(0, 1, size=(n_j, 1))  # biases for the hidden layer linear transform
        # self.B2 = np.random.uniform(0, 1, size=(n_k, 1))  # biases for the last layer linear transform

        # gradients; provides space for batch averaging
        # self.d_W1 = np.zeros((self.i, self.j, self.n_b))      # weights between the input and the hidden layer
        # self.d_W2 = np.zeros((self.j, self.k, self.n_b))      # weights between the hidden layer and the final layer
        # self.d_B1 = np.zeros(self.j, 1, self.n_b)             # biases for the hidden layer
        # self.d_B2 = np.zeros(self.k, 1, self.n_b)             # biases for the final layer

        self.LT1 = LinearTransform(self.i, self.j, self.n_b)
        self.ReLU = ReLU()
        self.LT2 = LinearTransform(self.j, self.k, self.n_b)
        self.SCC = SigmoidCrossEntropy(self.k)

    def train(self, x_batch, y_batch, LR, M, L2, number_in_batch):
        S1 = self.LT1.forward(x_batch)
        Z1 = self.ReLU.forward(S1)
        S2 = self.LT2.forward(Z1)
        E = self.SCC.forward(S2, y_batch)
        dE_dS2 = self.SCC.backward()
        dE_dZ1 = self.LT2.backward(dE_dS2, number_in_batch, LR, M, L2) # will update higher level of weights and biases
        dE_dS1 = self.ReLU.backward(dE_dZ1)
        dE_dX = self.LT1.backward(dE_dS1, number_in_batch, LR, M, L2)  # will update higher level of weights and biases

    def update_weights(self):
        # this is where wieghts and biases are actually updated, after the periodic delay. the found changes in weight
            # will be averaged and that will be this set of weight updates
        self.LT2.update_weights()
        self.LT1.update_weights()


    def evaluate(self, x, y):
        dud = 47


if __name__ == '__main__':

    ###################################
    ## Gather training and testing data
    ###################################
    # if sys.version_info[0] < 3:
    # data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    # else:
    #     data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')
    #
    # train_x = data['train_data']
    # train_y = data['train_labels']
    # test_x = data['test_data']
    # test_y = data['test_labels']

    train_x = np.array([[5, 10], [1, 3]])
    train_y = np.array([[6], [2]])


    ####################
    # initialize weights
    ####################
    # n_train = train_x.shape[0]                  # the number of training examples inputs, num_examples
    # n_test = test_x.shape[0]                    # the number of testing examples inputs
    # n_i = train_x.shape[1]                      # the number ot training inputs for each training example, input_dims
    # n_k = train_y.shape[1]                      # the number of possible training labels

    n_train = 2                                     # the number of training examples inputs
    n_test = 2                                      # the number of testing examples inputs
    n_i = 2                                         # the number ot training inputs for each training example
    n_k = 1                                         # the number of possible labels

    ########################
    # parameters that we set
    ########################
    n_j = 3                                 # number of nodes in the hidden layer
    num_epochs = 1000                       # number of time we will run through training set
    num_batches = 10                        # out delay in updating the weights, wait num_batch times before updating
    LR = 0.001                              # Learning Rate
    M = 0.9                                 # Momentum coefficient
    L2 = 1                                  # L2 penalty???

    mlp = MLP(n_i, n_j, n_k, num_batches)                                    # input_dims, hidden layer_dims, output_dims

    for epoch in range(num_epochs):                     # we only want to train on this set a certain amount of time
        number_in_batch = 0
        for example in range(n_train):                  # go through each example
            mlp.train(train_x[example], train_y[example], LR, M, L2, number_in_batch)   # x_batch, y_batch, learning_rate, momentum, l2_penalty
            if example/num_batches:                     # periodicly update weights
                mlp.update_weights()
                nummber_in_batch = 0
            number_in_batch += 1
    #         total_loss = 0.0
    #         # INSERT YOUR CODE FOR EACH MINI_BATCH HERE
	# 		# MAKE SURE TO UPDATE total_loss
    #         print('\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(epoch + 1, b + 1, total_loss), end='')
    #         sys.stdout.flush()
    #     # INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
	# 	# MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
    #     print()
    #     print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(train_loss, 100. * train_accuracy))
    #     print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(test_loss, 100. * test_accuracy))
