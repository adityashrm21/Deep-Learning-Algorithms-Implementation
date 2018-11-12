from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

class NeuralNet():
    '''
    This class contains methods/functions for the basic implementation
    of a neural network from scratch. The network will have an input layer
    with 3 nodes (excluding the bias unit), one hidden layer with 3 nodes
    and an output layer with one node.
    '''

    def __init__(self, W1, b1, W2, b2, X, y, cost_vec):

        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.X = X
        self.y = y
        self.cost_vec = cost_vec

    def sigmoid(self, t):

        return 1 / (1 + np.exp(-t))

    def cost_funtion(self, out, reg_lambda):

        cost = np.mean(np.square(out - self.y))
        cost += (reg_lambda / 2) * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return cost

    def predict(self, XX):

        a1 = XX
        z2 = np.dot(a1, self.W1) + self.b1.T
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.W2) + self.b2.T
        a3 = self.sigmoid(z3)
        print("Scaled input given for prediction: {0}".format(XX))
        print("Actual value for prediction: {0}".format(self.y))
        print("Scaled result of prediction: {0}".format(a3))

    def build_network(self, epochs = 1000, alpha = 0.001, reg_lambda = 0.01, print_verbose = False):

        # steps for gradient descent
        for epoch in range(epochs):

            delW1 = np.zeros(self.W1.shape)
            delW2 = np.zeros(self.W2.shape)
            delb1 = np.zeros(self.b1.shape)
            delb2 = np.zeros(self.b2.shape)

            # feedforward pass
            a1 = self.X
            #print("Shape of a1: {0}".format(a1.shape))
            z2 = np.dot(a1, self.W1) + self.b1.T
            #print("Shape of z2: {0}".format(z2.shape))
            a2 = self.sigmoid(z2)
            #print("Shape of a2: {0}".format(a2.shape))
            z3 = np.dot(a2, self.W2) + self.b2.T
            #print("Shape of z3: {0}".format(z3.shape))
            a3 = self.sigmoid(z3)
            #print("Shape of a3: {0}".format(a3.shape))
            out = a3

            #backpropagation
            del3 = (out - self.y) * (a3 *(1 - a3))
            #print("Shape of del3: {0}".format(del3.shape))
            del2 = np.dot(del3, self.W2.T) * (a2 *(1 - a2))
            #print("Shape of del2: {0}".format(del2.shape))

            delJ_W2 = np.dot(a2.T, del3)
            #print("Shape of delJ_W2: {0}".format(delJ_W2.shape))
            delJ_b2 = np.sum(del3, axis = 0, keepdims = True)
            #print("Shape of delJ_b2: {0}".format(delJ_b2.shape))
            delJ_W1 = np.dot(a1.T, del2)
            #print("Shape of delJ_W1: {0}".format(delJ_W1.shape))
            delJ_b1 = np.sum(del2, axis = 0, keepdims = True)
            #print("Shape of delJ_b1: {0}".format(delJ_b1.shape))

            delW1 += delJ_W1
            delb1 += delJ_b1.T
            delW2 += delJ_W2
            delb2 += delJ_b2.T

            # recalculating weights
            m = self.X.shape[0]
            self.W1 += -alpha * ((1 / m * delW1) + reg_lambda * self.W1)
            self.b1 += -alpha * (1 / m * delb1)
            self.W2 += -alpha * ((1 / m * delW2) + reg_lambda * self.W2)
            self.b2 += -alpha * (1 / m * delb2)

            curr_cost = self.cost_funtion(out, reg_lambda)
            self.cost_vec[0, epoch] = curr_cost
            if print_verbose:
                print("Iteration: {0}, Loss: {1}".format(epoch, self.cost_vec[0, epoch]))


def main():

    # Each row is a training example, each column is a feature  [X1,..., X5]
    # X = (hours studying, hours sleeping), y = score on test
    X = np.array(([2, 9], [1, 5], [3, 6], [5,8], [6,8]), dtype=float)
    y = np.array(([75], [62], [84], [90], [95]), dtype=float)

    # new input for prediction
    XX = np.array(([4, 7]), dtype = float)

    # scale units
    X = (X - np.mean(X, axis = 0))/ np.std(X, axis=0)
    XX = (XX - np.mean(XX, axis = 0))/ np.std(XX, axis=0)
    y = y / 100
    num_hidden = 3

    np.random.seed(0)
    W1 = np.random.randn(X.shape[1], num_hidden) / np.sqrt(X.shape[1])
    b1 = np.random.randn(num_hidden, 1)
    #b1 = np.zeros((num_hidden, 1))
    W2 = np.random.randn(num_hidden, y.shape[1]) / np.sqrt(num_hidden)
    b2 = np.random.randn(y.shape[1], 1)
    #b2 = np.zeros((y.shape[1], 1))

    epochs = 200000
    cost_vec = np.zeros((1, epochs))

    nn = NeuralNet(W1, b1, W2, b2, X, y, cost_vec)
    nn.build_network(epochs = epochs)

    nn.predict(X)

    plt.plot(range(epochs), cost_vec[0, :])
    plt.show()

if __name__ == '__main__':
    main()
