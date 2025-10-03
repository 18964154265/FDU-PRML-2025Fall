"""
criterion
"""

import math

def _entropy(counts, total):
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent

def _gini(counts, total):
    if total == 0:
        return 0.0
    s = 0.0
    for c in counts.values():
        p = c / total
        s += p * p
    return 1.0 - s

def _error(counts, total):
    if total == 0:
        return 0.0
    max_cnt = max(counts.values()) if counts else 0
    return 1.0 - (max_cnt / total)


def get_criterion_function(criterion):
    if criterion == "info_gain":
        return __info_gain
    elif criterion == "info_gain_ratio":
        return __info_gain_ratio
    elif criterion == "gini":
        return __gini_index
    elif criterion == "error_rate":
        return __error_rate


def __label_stat(y, l_y, r_y):
    """Count the number of labels of nodes"""
    left_labels = {}
    right_labels = {}
    all_labels = {}
    for t in y.reshape(-1):
        if t not in all_labels:
            all_labels[t] = 0
        all_labels[t] += 1
    for t in l_y.reshape(-1):
        if t not in left_labels:
            left_labels[t] = 0
        left_labels[t] += 1
    for t in r_y.reshape(-1):
        if t not in right_labels:
            right_labels[t] = 0
        right_labels[t] += 1

    return all_labels, left_labels, right_labels


def __info_gain(y, l_y, r_y):
    """
    Calculate the info gain

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain if splitting y into      #
    # l_y and r_y                                                             #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    n = len(y)
    nL = len(l_y)
    nR = len(r_y)

    # 无效分裂（某一侧为空），通常不给分
    if n == 0 or nL == 0 or nR == 0:
        return 0.0

    H_parent = _entropy(all_labels, n)
    H_left = _entropy(left_labels, nL)
    H_right = _entropy(right_labels, nR)

    info_gain = H_parent - (nL / n) * H_left - (nR / n) * H_right
   
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return info_gain
   


def __info_gain_ratio(y, l_y, r_y):
    """
    Calculate the info gain ratio

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    info_gain = __info_gain(y, l_y, r_y)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain ratio if splitting y     #
    # into l_y and r_y                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    n = len(y)
    nL = len(l_y)
    nR = len(r_y)

    if n == 0 or nL == 0 or nR == 0:
        return 0.0

    # SplitInfo
    split_info = 0.0
    for m in (nL, nR):
        p = m / n
        if p > 0:
            split_info -= p * math.log2(p)

    if split_info == 0.0:
        return 0.0
    return info_gain / split_info
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    


def __gini_index(y, l_y, r_y):
    """
    Calculate the gini index

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the gini index value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    n = len(y)
    nL = len(l_y)
    nR = len(r_y)

    if n == 0 or nL == 0 or nR == 0:
        return 0.0

    G_parent = _gini(all_labels, n)
    G_left = _gini(left_labels, nL)
    G_right = _gini(right_labels, nR)

    before = G_parent
    after = (nL / n) * G_left + (nR / n) * G_right
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after


def __error_rate(y, l_y, r_y):
    """Calculate the error rate"""
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the error rate value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    n = len(y)
    nL = len(l_y)
    nR = len(r_y)

    if n == 0 or nL == 0 or nR == 0:
        return 0.0

    E_parent = _error(all_labels, n)
    E_left = _error(left_labels, nL)
    E_right = _error(right_labels, nR)

    before = E_parent
    after = (nL / n) * E_left + (nR / n) * E_right
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after
