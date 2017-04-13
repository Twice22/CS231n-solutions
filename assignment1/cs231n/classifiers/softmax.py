import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    s_ = X[i].dot(W)

    # substracting all value by the same value won't change the softmax
    s_ -= np.max(s_)

    s_yi = s_[y[i]]
    s_j = np.sum(np.exp(s_))
    loss += -s_yi + np.log(s_j)

    # computing the gradient
    for k in xrange(num_classes):
      s_k = np.exp(s_[k])/s_j
      dW[:, k] += (s_k - (k == y[i])) * X[i]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  num_classes = W.shape[1] #C
  num_train = X.shape[0] #N
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X.dot(W) #(N, C)
  f -= np.max(f, axis=1, keepdims=True) #for stability

  # Loss: L_i = - f(x_i)_{y_i} + log \sum_j e^{f(x_i)_j}
  # Compute vector of stacked correct f-scores: [f(x_1)_{y_1}, ..., f(x_N)_{y_N}]
  # (where N = num_train)
  f_y = f[range(num_train), y]

  exp_f_y = np.exp(f_y)
  exp_f = np.exp(f)

  loss = -np.sum(np.log(exp_f_y/np.sum(exp_f, axis=1)))

  #compute gradient
  p = exp_f/np.sum(exp_f, axis=1, keepdims=True)
  ind = np.zeros_like(f)
  ind[range(num_train),  y] = 1
  dW = X.T.dot(p - ind)

  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

