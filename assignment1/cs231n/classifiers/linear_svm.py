import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      # Both W and dW is a matrix of shape (D,C), so W can be written as (wc1, wc2, ..., WcC) and dW can be written as (dwc1, dwc2, ..., dwcC). Here, scores[j] only relates with wcj and corrrect_class only relates with wcyi. So we can only add the gradient of these 2 column vectors. 
      d_margin_score_j = np.zeros(W.shape)
      # Equals to X.T * (A = zeroes((N,C)), A[i][j]=1)
      d_margin_score_j[:, j] = X[i].T
      d_margin_score_correct = np.zeros(W.shape)
      # Equals to X.T * (A = zeroes((N,C)), A[i][y[i]]=1)
      d_margin_score_correct[:, y[i]] = X[i].T
      if margin > 0:
        loss += margin
        dW += d_margin_score_j - d_margin_score_correct  
            
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  # np.sum(W*W) is the same as np.linalg.norm(wc1) + ... + np.linalg.norm(wcC), so its gradient is (dwc1, dwc2, ..., dwcC) = 2*W  
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)  # (N, C)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  y_scores = [scores[index, classIdx] for index, classIdx in enumerate(y)]
  y_scores_v = np.vstack(y_scores)
  sub_target_score = np.ones((num_train, num_classes)) * y_scores_v
  margin = scores - sub_target_score + 1 
  greater_zero_matrix = np.array(margin > 0, dtype=int)   #(N, C)
  margin = margin * greater_zero_matrix
  loss = np.sum(margin)/num_train - 1
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # We can resort to the equally matrix calculation in naive method to        #
  # implement vectorized version here
  # For the j == y[i] position, in fact it doesn't affect the gradient(zero), #
  # so can be caculated too to make the algorithm simpler                     #
  # N*1, means how many times do we need class y[i]
  cls_sum = np.sum(greater_zero_matrix, axis=1)    
  A = np.zeros((num_train, num_classes))
  for idx, cls in enumerate(y):
    A[idx][cls] = cls_sum[idx]
  dW = X.T.dot(greater_zero_matrix - A)/num_train
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
