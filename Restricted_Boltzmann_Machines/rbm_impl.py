from __future__ import print_function
import numpy as np

class RbmImpl:
    '''
    This class implements Restricted Boltzman Machines
    '''

    def __init__(self, num_visible, num_hidden):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.verbose = True
        np_rng = np.random.RandomState(3412)

        self.weights = np.asarray(np_rng.uniform(
                    low=-4 * np.sqrt(6. / (num_hidden + num_visible)),
                    high=4 * np.sqrt(6. / (num_hidden + num_visible)),
                    size=(num_visible, num_hidden)))

        self.weights = np.insert(self.weights, 0, 0, axis = 0)
        self.weights = np.insert(self.weights, 0, 0, axis = 1)


    def train_rbm(self, data, max_epochs = 2000, learning_rate = 0.08):

        num_examples = data.shape[0]
        data = np.insert(data, 0, 1, axis = 1)

        for epoch in range(max_epochs):
            pos_hid_activations = np.dot(data, self.weights)
            pos_hid_probs = self.sigmoid(pos_hid_activations)
            pos_hid_probs[:,0] = 1
            pos_hid_states = pos_hid_probs > np.random.rand(num_examples,
                            self.num_hidden + 1)
            pos_associations = np.dot(data.T, pos_hid_probs)


            neg_vis_activations = np.dot(pos_hid_states, self.weights.T)
            neg_vis_probs = self.sigmoid(neg_vis_activations)
            neg_vis_probs[:,0] = 1

            neg_hid_activations = np.dot(neg_vis_probs, self.weights)
            neg_hid_probs = self.sigmoid(neg_hid_activations)
            neg_associations = np.dot(neg_vis_probs.T, neg_hid_probs)

            self.weights += learning_rate * ((pos_associations -
                            neg_associations) / num_examples)
            error = np.sum((data - neg_vis_probs) ** 2)
            if self.verbose:
                print('Epoch %s: Error is: %s', (epoch, error))



    def sigmoid(self, val):
        return 1.0 / (1 + np.exp(-val))

if __name__ == '__main__':
    rbmInstance = RbmImpl(num_visible = 6, num_hidden = 2)
    training_data = np.array([[1,1,1,0,0,0], [1,0,1,0,0,0], [1,1,1,0,0,0],
                    [0,0,1,1,1,0], [0,0,1,1,0,0], [0,0,1,1,1,0]])
    rbmInstance.train_rbm(data = training_data, max_epochs = 5000)
    print('The weights obtained after training are:')
    print(rbmInstance.weights)
