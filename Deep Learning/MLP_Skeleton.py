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
from math import e
import matplotlib.pyplot as plt

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
        self.W = np.random.uniform(0, .001, size=(self.i, self.j))  # weights that go from the input into this layer
        self.B = np.random.uniform(0, .01, size=(self.j, 1))      # biases for the hidden layer linear transform
        self.d_B = np.zeros([self.j, 1, self.n_b])
        self.m_W = np.zeros([self.i, self.j])                   # the last momentum terms for W
        self.m_B = np.zeros([self.j, 1])
        self.S = np.zeros([self.j, 1])  # output vector from linear transform

    def forward(self, x):
        # x is [ix1], w is [ixj], b is [1xj], Z = [jx1]
        self.x = np.reshape(x, [self.i, 1])                          # need to remember the input for back propagation
        self.S = np.reshape(np.dot(x.transpose(), self.W), (self.j, 1))                 # x_col dot with W row
        self.S = np.add(self.S, self.B)                                                 # add biases
        return self.S

    def backward(self, grad_output, b, LR=0.0):
        """ grad_output is the delta calulcated before accounting for this last step of linearization
            learning_rate is the amount we are allowing the gradient to correct our weights
            b is the batch number"""
        self.d_B[:, :, b] = np.multiply(grad_output[:], LR)
        dE_dW = np.reshape(np.dot(self.x[:], grad_output[:].transpose()), (self.i, self.j))        # the base gradients
        plane_of_gradients = np.multiply(dE_dW, LR)                        # -LR * dE/dW

        try:            # add this new column of gradients for momentum purposes
            self.d_W = np.append(self.G, plane_of_gradients, axis=2)
        except:
            self.d_W = np.atleast_3d(plane_of_gradients)              # creating of the 3D gradient list for momentum

        return dE_dW

    def calculate_momentum_values(self, M, avg_d_B, avg_d_W):
        # will apply momentum to the gradients to provide one final gradient that this round of weight updates will be
        self.m_B = np.multiply(self.m_B, M) + avg_d_B
        self.m_W = np.multiply(self.m_W, M) + avg_d_W

    def update_weights(self, M = 0.0, L2 = 0.0):
        # call this to do the work of updating the weights and biases
        avg_d_B = np.mean(self.d_B, axis=2)         # used to find the average of the gradients within this batch
        avg_d_W = np.mean(self.d_W, axis=2)

        self.calculate_momentum_values(M, avg_d_B, avg_d_W)

        # actual weight update
        self.B = np.add(self.B, self.m_B)
        self.W = np.add(self.W, self.m_W)


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
        self.n_correct = 0              # number of times the system calculatd correctly

    def forward(self, s, y):
        """ s is the sum found through the linear transformation that occurs right before this step
            y is the target values for this example"""
        self.y = y
        e_array = np.full((self.k, 1), e)
        self.z_SCC = np.divide(1, np.add(1, np.power(e_array, np.multiply(-1, s))))     # sigmoid
        cross_A = np.multiply(y, np.log(self.z_SCC))
        cross_B = np.multiply(np.subtract(1, y), np.log(np.subtract(1, self.z_SCC)))
        self.E_SCC = np.multiply(-1, np.add(cross_A, cross_B))                          # cross entropy
        predicted_answer = np.around(self.E_SCC)                                # as this is binary, this will be 0 or 1

        # check the accuracy of the system
        if predicted_answer != self.y:              # low entropy means we are close to the right answer
            self.n_correct += 1

        return self.E_SCC

    def backward(self):
        # dE/dz = -y*/z + (1-y*)/(1-z); dz/ds = z*(1-z)
        return np.subtract(self.z_SCC, self.y)          # the delta based off of the loss function and sigmoid

    def accuracy(self, n_examples):
        # calculate the accuracy of this epoch; accuracy = n_correct_predictions/n_total_examples
        return self.n_correct / n_examples


class MLP(object):
    # This is a class for the Multilayer perceptron
    def __init__(self, input_dims, hidden_units, num_outputs, n_batch):
        self.i = input_dims                         # [i x 1]
        self.j = hidden_units                       # [jx1]
        self.k = num_outputs
        self.n_b = n_batch                          # number of time we calculate the deltas before averaging and actually applying

        self.LT1 = LinearTransform(self.i, self.j, self.n_b)
        self.ReLU = ReLU()
        self.LT2 = LinearTransform(self.j, self.k, self.n_b)
        self.SCC = SigmoidCrossEntropy(self.k)

    def train(self, x_batch, y_batch, LR, number_in_batch):
        S1 = self.LT1.forward(x_batch)
        Z1 = self.ReLU.forward(S1)
        S2 = self.LT2.forward(Z1)
        E = self.SCC.forward(S2, y_batch)
        dE_dS2 = self.SCC.backward()
        dE_dZ1 = self.LT2.backward(dE_dS2, number_in_batch, LR)     # will update higher level of weights and biases
        dE_dS1 = self.ReLU.backward(dE_dZ1)
        dE_dX = self.LT1.backward(dE_dS1, number_in_batch, LR)      # will update higher level of weights and biases

        return E

    def test(self, x_batch, y_batch):
        S1 = self.LT1.forward(x_batch)
        Z1 = self.ReLU.forward(S1)
        S2 = self.LT2.forward(Z1)
        E = self.SCC.forward(S2, y_batch)

        return E

    def update_weights(self, M, L2):            #### FROZE Weights 1
        # this is where weights and biases are actually updated, after the periodic delay
        self.LT2.update_weights(M, L2)
        # self.LT1.update_weights(M, L2)

    def evaluate(self, examples_in_epoch):
        return self.SCC.accuracy(examples_in_epoch)


def lmd_plot(x, y, x_label, y_label, title):            # save_plot, path):
    # this function just plots the basics and includes labels and such
    plt.plot(x, y)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    # if save_plot:
    #     plt.savefig(path + title + ".png")
    # plt.close()
    plt.show()


def Normalize_inputs(data):
    # make sure that the input is from 0 to 1
    max_val = 255
    norm = np.true_divide(data, max_val)
    return norm


if __name__ == '__main__':

    ##################################
    # Gather training and testing data
    ##################################
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')

    train_x = data['train_data']
    train_x = Normalize_inputs(train_x)
    train_y = data['train_labels']
    test_x = data['test_data']
    test_x = Normalize_inputs(test_x)
    test_y = data['test_labels']

    # train_x = np.array([[5, 10], [1, 3]])
    # train_x = Normalize_inputs(train_x)
    # train_y = np.array([[6], [2]])


    ####################
    # initialize weights
    ####################
    n_train = train_x.shape[0]                  # the number of training examples inputs, num_examples
    n_test = test_x.shape[0]                    # the number of testing examples inputs
    n_i = train_x.shape[1]                      # the number ot training inputs for each training example, input_dims
    n_k = train_y.shape[1]                      # the number of possible training labels

    # n_train = 2                                     # the number of training examples inputs
    # n_test = 2                                      # the number of testing examples inputs
    # n_i = 2                                         # the number ot training inputs for each training example
    # n_k = 1                                         # the number of possible labels

    ########################
    # parameters that we set
    ########################
    n_j = 10                                 # number of nodes in the hidden layer
    num_epochs = 1000                       # number of time we will run through training set
    num_batches = 100                        # out delay in updating the weights, wait num_batch times before updating
    LR = 0.001                              # Learning Rate
    M = 0.8                                 # Momentum coefficient
    L2 = 1                                  # L2 penalty???

    ################
    # Data Storage
    ################
    train_loss_list_per_epoch = []
    train_loss_list_overall = []
    train_accuracy_list = []

    for epoch in range(num_epochs):                     # we only want to train on this set a certain amount of time
        mlp = MLP(n_i, n_j, n_k, num_batches)           # input_dims, hidden layer_dims, output_dim
        number_in_batch = 0
        total_loss = 0.0
        mb= 0
        for example in range(n_train):
            # for example in range(n_train):                  # go through each example
            E = mlp.train(train_x[example], train_y[example], LR, number_in_batch)   # x_batch, y_batch, learning_rate, momentum, l2_penalty
            if example % (num_batches-1) == 0 and example != 0:                     # periodicly update weights
                mlp.update_weights(M, L2)
                number_in_batch = 0
                average_loss = np.mean(train_loss_list_per_epoch)
                print('\r[Epoch {}, mb {}]    Avg.Loss = {}'.format(epoch + 1, mb, round(average_loss, 5)), end='')
                mb += 1
            number_in_batch += 1
            train_loss_list_per_epoch.append(E[0])

        train_loss_list_overall.append(average_loss)
        train_accuracy = mlp.evaluate(n_train)

        print('\r[Epoch {} Final]    Avg.Loss = {}'.format(epoch + 1, round(total_loss, 5)), end='')
        sys.stdout.flush()
    # 	# MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
    print()
    print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(average_loss, 100. * train_accuracy))
    #     print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(test_loss, 100. * test_accuracy))

    ##############
    # Plot Time!
    ##############
    list_o_epochs = np.linspace(0, num_epochs, num_epochs)
    lmd_plot(list_o_epochs, train_loss_list_overall, "Epochs", "Loss", "Training Loss")

