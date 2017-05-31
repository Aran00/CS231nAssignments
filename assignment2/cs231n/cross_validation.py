import numpy as np
from cs231n.solver import Solver


solver_config_temp = {
    "num_epochs": 20,
    "batch_size": 1000,
    "update_rule": 'sgd_momentum',
    "optim_config": {
        'learning_rate': 1e-2,
    },
    "lr_decay": 0.95,
    "verbose": True,
    "print_every": 10
}


def cross_val_score(model, solver_config, X, y, cv=5):
    '''
    # Make cv just be an integer here
    For a neural network model, solver also has some hyperparams, but we don't verify them here
    '''
    num_folds = cv

    X_train_folds = []
    y_train_folds = []
    ################################################################################
    # TODO:                                                                        #
    # Split up the training data into folds. After splitting, X_train_folds and    #
    # y_train_folds should each be lists of length num_folds, where                #
    # y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
    # Hint: Look up the numpy array_split function.                                #
    ################################################################################
    # permutated_index = np.random.permutation(num_training)
    num_X = X.shape[0]
    folders_index = np.array_split(range(num_X), num_folds)  # A list of nd array
    for index_arr in folders_index:
        X_train_folds.append(X[index_arr])
        y_train_folds.append(y[index_arr])
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################


    ################################################################################
    # TODO:                                                                        #
    # Perform k-fold cross validation and get the average score.                   #
    ################################################################################

    val_acc_folds = []
    for fold_idx in xrange(num_folds):
        X_train_list = X_train_folds[:fold_idx] + X_train_folds[fold_idx + 1:]
        y_train_list = y_train_folds[:fold_idx] + y_train_folds[fold_idx + 1:]
        X_train_fold = np.concatenate(X_train_list)  # Or np.vstack
        y_train_fold = np.concatenate(y_train_list)  # Or np.hstack
        X_validation_fold = X_train_folds[fold_idx]
        y_validation_fold = y_train_folds[fold_idx]
        data = {
            'X_train': X_train_fold,
            'y_train': y_train_fold,
            'X_val': X_validation_fold,
            'y_val': y_validation_fold
        }
        solver = Solver(model, data, **solver_config)
        solver.train()
        val_acc_folds.append(solver.best_val_acc)

    return np.average(val_acc_folds)

