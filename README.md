# (Deep Learning + Machine Learning) Algorithms Implementation
This repository contains implementation of various deep learning and machine learning algorithms in Python. Different library/packages will be used for the implementations like TensorFlow, Keras, PyTorch, Scikit-Learn, etc.
_Note_: I am regularly adding new implementations to the repository.

## Multilayer Perceptron

1. The jupyter notebook [neural_net_from_scratch.ipynb](https://github.com/adityashrm21/Deep-Learning-Algorithms-Implementation/blob/master/Multilayer_Perceptron/neural_net_from_scratch.ipynb) contains the raw implementation in `python` for a basic Multilayer Perceptron.
2. The `python` script [neural_net_from_scratch.py](https://github.com/adityashrm21/Deep-Learning-Algorithms-Implementation/blob/master/Multilayer_Perceptron/neural_net_from_scratch.py) is a cleaner version of the notebook version.
3. The jupyter notebook [tensorflow_mlp_regression.ipynb](https://github.com/adityashrm21/Deep-Learning-Algorithms-Implementation/blob/master/Multilayer_Perceptron/tensorflow_mlp_regression.ipynb) contains a `tensorFlow` implementation of a Multilayer Perceptron with an example using regression on California housing dataset from the `scikit-learn` library.
4. The jupyter notebook [tensorflow_mlp_classification.ipynb](https://github.com/adityashrm21/Deep-Learning-Algorithms-Implementation/blob/master/Multilayer_Perceptron/tensorflow_mlp_classification.ipynb) contains a tensorFlow implementation of a Multilayer Perceptron with an example using classification on the MNIST dataset.

## Restricted Boltzmann Machine

1. The script [rbm_impl.py](https://github.com/adityashrm21/Deep-Learning-Algorithms-Implementation/blob/master/Restricted_Boltzmann_Machines/rbm_impl.py) contains a basic implementation of a restricted boltzmann machine in `python`.

## Linear Regression

1. The jupyter notebook [linear_regression_scratch.ipynb](https://github.com/adityashrm21/Deep-Learning-Algorithms-Implementation/blob/master/Linear_Regression/linear_regression_scratch.ipynb) contains basic implementation of linear regression in python using `numpy` on a randomly generated dataset. The optimization is done using full-batch gradient descent.

## Logistic Regression

1. The jupyter notebook [logistic_regression_scratch.ipynb](https://github.com/adityashrm21/Deep-Learning-Algorithms-Implementation/blob/master/Logistic_Regression/logistic_regression_scratch.ipynb) contains basic implementation of logistic regression in python using `numpy` using the iris dataset. Two of the linearly inseparable species are combined together into one category. The optimization is done using full-batch gradient descent.

## Decision Trees

1. The jupyter notebook [decision_trees_sklearn.ipynb](https://github.com/adityashrm21/Deep-Learning-Algorithms-Implementation/blob/master/Decision_Trees/decision_trees_sklearn.ipynb) contains an implementation of the decision tree algorithm in Python using scikit-learn. The iris dataset is used to make a prediction using decision trees.
2. The jupyter notebook [decision_trees_scratch.ipynb](https://github.com/adityashrm21/Deep-Learning-Algorithms-Implementation/blob/master/Decision_Trees/decision_tree_scratch.ipynb) contains an implementation of the decision tree algorithm in Python from scratch using numpy. The [Spotify song attribute](https://www.kaggle.com/geomack/spotifyclassification/home) dataset is used to make a prediction using the decision trees and compared with the scikit-learn implementation.


Sources:
- [Edwin Chen's Blog on introduction to RBMs](http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/)
- [Denny Britz's tutorial on writing neural networks from scratch](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/)
- [Samay Shamdasani's tutorial on building a neural network](https://enlight.nyc/projects/neural-network/)
- [Official TensorFlow documentation and tutorials](https://www.tensorflow.org/tutorials/)
