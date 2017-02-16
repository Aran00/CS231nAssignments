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
  
  See the loss function in Lecture 3.
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
  num_dim = X.shape[1]
  for i in xrange(num_train):
    scores = X[i].dot(W)  # 1*C or (C,)
    # As when scores[i] is too large, the exp result would overflow, we need to shift this result. It would not change the final loss.
    scores -= np.max(scores)
    exp_scores = np.exp(scores) # (C, )
    exp_sum = np.sum(exp_scores)
    loss_i = (-1.0) * (scores[y[i]] - np.log(exp_sum))  
    d_loss_i_first = np.zeros(W.shape)
    d_loss_i_first[:, y[i]] = X[i].T  
    d_loss_i_second = 1.0/exp_sum * exp_scores * np.dot(X[i].reshape(num_dim, 1), np.ones((1, num_classes)))
    loss += loss_i
    dW += (-1.0) * (d_loss_i_first - d_loss_i_second)                   
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(np.power(W, 2)) 
  dW += reg * W                     
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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  scores = X.dot(W)    # N * C  
  scores -= np.vstack(np.max(scores, axis=1))
  exp_scores = np.exp(scores) # N * C
  y_scores = [scores[index, classIdx] for index, classIdx in enumerate(y)]
  # y_scores_v = np.vstack(y_scores)
  
  exp_sum = np.sum(exp_scores, axis=1)   # (N,)
  loss = np.mean(np.log(exp_sum) - y_scores)
    
  A = np.zeros((num_train, num_classes))
  for idx, cls in enumerate(y):
    A[idx][cls] = 1  
  dW_first = X.T.dot(A)
  normalized_exp_scores = 1.0/np.vstack(exp_sum) * exp_scores 
  dW_second = np.dot(X.T, normalized_exp_scores)
  dW = (-1.0) * (dW_first - dW_second) / num_train  

  # Add regularization to the loss.
  loss += reg * np.sum(np.power(W, 2))/2
  # np.sum(W*W) is the same as np.linalg.norm(wc1) + ... + np.linalg.norm(wcC), so its gradient is (dwc1, dwc2, ..., dwcC) = 2*W  
  dW += reg * W  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW

