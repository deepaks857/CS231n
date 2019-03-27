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
  N = X.shape[0]
  C = W.shape[1]
  for i in range(N):
      scores = X[i].dot(W)
      f = scores - np.max(scores)
      denom = np.sum(np.exp(f))
      loss += - np.log(np.exp(f[y[i]])/denom)
      for j in range(C):
          if j==y[i]:
              dW[:,j] += X[i] * ( (np.exp(f[y[i]]) / denom ) - 1 )
          else:
              dW[:,j] += X[i] * (np.exp(f[j]) / denom)
  loss /= N
  dW /= N
  loss +=  0.5*reg * np.sum(W*W)
  dW +=  reg*(W)
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
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W) # (N, C)
  scores -= np.amax(scores, axis=1)[:, None]
  loss = np.mean(-scores[np.arange(scores.shape[0]), y] + (np.log(np.sum(np.exp(scores), axis=1))) )
  #multiplier2 = (np.exp(scores))[np.arange(scores.shape[0]), y] / np.sum( np.exp(scores), axis=1 )
  binary = np.zeros_like(scores)
  binary[np.arange(scores.shape[0]), y] = 1
  mult = (binary * ( np.exp(scores) / np.sum(np.exp(scores), axis=1)[:,None])) - binary
  binary2 = np.ones_like(binary)
  binary2 = binary2 - binary
  binary2 = binary2 * (np.exp(scores)/np.sum(np.exp(scores), axis=1)[:,None])
  dW += X.T.dot(mult)
  dW += X.T.dot(binary2)
  dW /= X.shape[0]
  loss += 0.5* reg * np.sum(W*W)
  dW += reg*(W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
