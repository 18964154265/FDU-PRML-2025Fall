import os
import numpy as np
from typing import List, Tuple
from data_generate import generate_and_save, load_prepared_dataset
from viz_knn import plot_k_curve, plot_decision_boundary_multi

# 输出目录
OUT_DIR = "./output"
DATA_DIR = "./input_knn"
FIG_K_CURVE   = os.path.join(OUT_DIR, "knn_k_curve.png")
FIG_BOUNDARY  = os.path.join(OUT_DIR, "knn_boundary.png")

# ============ TODO 1：pairwise_dist ============
def pairwise_dist(X_test, X_train, metric, mode):
    """
    Compute pairwise distances between X_test (Nte,D) and X_train (Ntr,D).

    Required:
      - L2 distance 'l2' with modes:
          * 'two_loops'  
          * 'no_loops' 
      - 'cosine' distance (distance = 1 - cosine_similarity)
    """
    X_test = np.asarray(X_test, dtype=np.float64)
    X_train = np.asarray(X_train, dtype=np.float64)
    Nte, D  = X_test.shape
    Ntr, D2 = X_train.shape
    assert D == D2, "Dim mismatch between test and train."

    if metric == "l2":
        if mode == "two_loops":
            # =============== TODO (students, REQUIRED) ===============
            dists = np.zeros((Nte, Ntr))
            for i in range(Nte):
                for j in range(Ntr):
                    diff = X_test[i] - X_train[j]
                    dists[i,j] = np.sqrt(np.sum(diff * diff))
            return dists
            # =========================================================
            # raise NotImplementedError("Implement L2 two_loops")

        elif mode == "no_loops":
            # =============== TODO (students, REQUIRED) ===============
            test_square = np.sum(X_test**2, axis=1, keepdims=True) 
            train_square = np.sum(X_train**2, axis=1)
            cross_term = np.dot(X_test, X_train.T)
            dists = np.sqrt(test_square + train_square - 2*cross_term)
            return dists
            # =========================================================
            # raise NotImplementedError("Implement L2 no_loops")

        else:
            raise ValueError("Unknown mode for L2.")

    elif metric == "cosine":
        # =============== TODO (students, REQUIRED) ===============
        test_norm = np.linalg.norm(X_test, axis=1, keepdims=True)  
        train_norm = np.linalg.norm(X_train, axis=1)               
        cosine_similarity = np.dot(X_test, X_train.T) / (test_norm * train_norm + 1e-12)
        dists = 1 - cosine_similarity
        return dists
        # ================================================
        # raise NotImplementedError("cosine distance")
    else:
        raise ValueError("metric must be 'l2' or 'cosine'.")


# ============ TODO 2：knn_predict（多数表决） ============
def knn_predict(X_test, X_train, y_train, k, metric, mode):
    """
    kNN prediction.
    Required: majority vote with L2 distance.

    Returns
    -------
    y_pred : (Nte,) int
    """
    dists = pairwise_dist(X_test, X_train, metric=metric, mode=mode)
    y_train = np.asarray(y_train).reshape(-1).astype(int)
    Nte = dists.shape[0]
    y_pred = np.zeros(Nte, dtype=int)

    for i in range(Nte):
        idx = np.argsort(dists[i])[:k]
        neighbors = y_train[idx]

        # =============== TODO (students, REQUIRED) ===============
        counts = np.bincount(neighbors)
        y_pred[i] = np.argmax(counts)
        # ===========================================
        # raise NotImplementedError("Implement majority vote in knn_predict")

    return y_pred


# ============ TODO 3：select_k_by_validation ============
def select_k_by_validation(X_train, y_train, X_val, y_val, ks: List[int], metric, mode) -> Tuple[int, List[float]]:
    """
    Grid-search K on validation set.

    Returns
    -------
    best_k : int
    accs   : list of validation accuracies aligned with ks
    """
    # =============== TODO (students, REQUIRED) ===============
    accs = []
    for k in ks:
        y_val_pred = knn_predict(X_val, X_train, y_train, k, metric, mode)
        accuracy = np.mean(y_val_pred == y_val)
        accs.append(accuracy)
    best_k = ks[np.argmax(accs)]
    return best_k, accs
    # =========================================================
    # raise NotImplementedError("Implement select_k_by_validation")


def run_with_visualization(n_classes, cluster_std, test_size, val_size):
    X_train, y_train, X_val, y_val, X_test, y_test = load_prepared_dataset(DATA_DIR)

    ks = [1, 3, 5, 7, 9, 11, 13]
    metric = "l2"           # ["l2", "cosine"]
    mode   = "no_loops"     # ["two_loops", "no_loops", "one_loop"]

    best_k, accs = select_k_by_validation(X_train, y_train, X_val, y_val,
                                          ks, metric=metric, mode=mode)
    if(max(accs)<0.95):
        return 
    print(f"Generating data with N_CLASSES={N_CLASSES}, CLUSTER_STD={CLUSTER_STD}, TEST_SIZE={TEST_SIZE}, VAL_SIZE={VAL_SIZE}")
    print(f"[ModelSelect] best k={best_k} (val acc={max(accs):.4f})")
    plot_k_curve(ks, accs, os.path.join(OUT_DIR, "knn_k_curve.png"))

    X_trv = np.vstack([X_train, X_val]); y_trv = np.hstack([y_train, y_val])
    def predict_fn_for_k(k):
        return lambda Xq: knn_predict(Xq, X_trv, y_trv, k, metric=metric, mode=mode)

    ks_panel = sorted(set(ks + [best_k]))
    '''plot_decision_boundary_multi(predict_fn_for_k, X_train, y_train, X_test, y_test,
                                 ks=ks_panel,
                                 out_path=os.path.join(OUT_DIR, "knn_boundary_grid.png"),
                                 grid_n=200, batch_size=4096)
'''
DATA_DIR     = "./input_knn"    # 输入数据目录
RANDOM_STATE = 42               # 随机种子
N_SAMPLES    = 1000              # 样本总数
N_CLASSES    = 4                # 类别数 （在boundary图上会有几个色块）
CLUSTER_STD  = 4.0              # 类内标准差（数据难度）
TEST_SIZE    = 0.25             # 测试集比例    
VAL_SIZE     = 0.25             # 验证集比例

for n_classes in [2,3, 4,5,6,7,8]:
    N_CLASSES = n_classes
    for cluster_std in [2.0, 2.5, 3.0, 3.5, 4.0,4.5,5.0,5.5,6.0]:
        CLUSTER_STD = cluster_std
        for(test_size, val_size) in [(0.25, 0.25), (0.2, 0.2), (0.15, 0.15),(0.1, 0.1)]:
            TEST_SIZE = test_size
            VAL_SIZE = val_size
            
            generate_and_save(
                data_dir = DATA_DIR,
                n_samples = 1000,
                n_classes = N_CLASSES,
                cluster_std = CLUSTER_STD,
                test_size = TEST_SIZE,
                val_size = VAL_SIZE,
                random_state = RANDOM_STATE,
                force = True,
            )
            run_with_visualization(N_CLASSES, CLUSTER_STD, TEST_SIZE, VAL_SIZE)

'''
if __name__ == "__main__":
    run_with_visualization()'''
