import numpy as np

def accuracy_score(y_true, y_pred):
    accuracy = -1
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    # =============== TODO (students) ===============
    tp = np.sum(y_true == y_pred)
    accuracy = tp / len(y_true) 
    return accuracy
    # ===============================================
    # raise NotImplementedError("Implement accuracy_score")


def mean_squared_error(y_true, y_pred):
    error = -1
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    # =============== TODO (students) ===============
    error = np.mean((y_true - y_pred) ** 2)
    return error
    # ===============================================
    # raise NotImplementedError("Implement mean_squared_error")
