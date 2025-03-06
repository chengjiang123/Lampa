import torch
import torch.nn.functional as F
import numpy as np
from numba import jit
from torch_scatter import scatter_mean
from sklearn.cluster import DBSCAN


def pair_filter(cluster_ids, point_pairs, recons, pts, pt_thres):
    # Have been taken care of when generating the dataset
    # non_zero_pid_point_pairs = (cluster_ids[point_pairs[0]] != 0) & (cluster_ids[point_pairs[1]] != 0)

    reconstructable_point_pairs = (recons[point_pairs[0]] != 0) & (recons[point_pairs[1]] != 0)
    high_pt_point_pairs = (pts[point_pairs[0]] > pt_thres) & (pts[point_pairs[1]] > pt_thres)
    mask = reconstructable_point_pairs & high_pt_point_pairs
    return mask


def point_filter(cluster_ids, recons, pts, pt_thres):
    mask = (cluster_ids != 0) & (recons != 0) & (pts > pt_thres)
    return mask


@torch.no_grad()
def acc_and_pr_at_k(embeddings, cluster_ids, mask, dist_metric='l2_rbf', K=19, batch_size=None, eps=0.3, min_samples=1, use_dbscan=False):
    cluster_ids = cluster_ids.cpu().numpy()
    mask = mask.cpu().numpy()

    num_points = embeddings.shape[0]
    if batch_size is None:
        batch_size = num_points

    unique_clusters, counts = np.unique(cluster_ids, return_counts=True)
    cluster_sizes = dict(zip(unique_clusters.tolist(), counts.tolist()))
    if use_dbscan:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='l2')
        dbscan_labels = dbscan.fit_predict(embeddings.cpu().numpy())  # Apply DBSCAN to embeddings

        dbscan_labels = np.where(dbscan_labels == -1, -2, dbscan_labels)

        unique_dbscan_clusters, dbscan_counts = np.unique(dbscan_labels, return_counts=True)
        dbscan_cluster_sizes = dict(zip(unique_dbscan_clusters.tolist(), dbscan_counts.tolist()))
    else:
        dbscan_labels = None
        dbscan_cluster_sizes = {}

    accuracy_scores_cluster = []
    precision_scores_cluster = []
    recall_scores_cluster = []
    efficiency_dm_scores_cluster = []  
    efficiency_lhc_scores_cluster = [] 

    accuracy_scores_dbscan = []
    precision_scores_dbscan = []
    recall_scores_dbscan = []
    efficiency_dm_scores_dbscan = [] 
    efficiency_lhc_scores_dbscan = [] 


    for start_index in range(0, num_points, batch_size):
        end_index = min(start_index + batch_size, num_points)
        batch_mask = mask[start_index:end_index]
        batch_embeddings = embeddings[start_index:end_index][batch_mask]
        batch_cluster_ids = cluster_ids[start_index:end_index][batch_mask]

 
        if use_dbscan:
            batch_dbscan_labels = dbscan_labels[start_index:end_index][batch_mask]
        else:
            batch_dbscan_labels = None

        if "l2" in dist_metric:
            dist_mat_batch = torch.cdist(batch_embeddings, embeddings, p=2.0)
        elif dist_metric == "cosine":
            dist_mat_batch = 1 - F.cosine_similarity(batch_embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        else:
            raise NotImplementedError

        k_list_cluster = np.array([cluster_sizes[each_cluster_id] - 1 for each_cluster_id in batch_cluster_ids])
        
        if use_dbscan:
            k_list_dbscan = np.array([dbscan_cluster_sizes[each_dbscan_id] - 1 for each_dbscan_id in batch_dbscan_labels])
        else:
            k_list_dbscan = []

        assert max(k_list_cluster) <= K, f"K is too small, max k is {max(k_list_cluster)}"
        if use_dbscan:
            K = max(k_list_dbscan) + 1
            assert max(k_list_dbscan) <= K, f"K is too small, max k for DBSCAN is {max(k_list_dbscan)}"

            indices = dist_mat_batch.topk(K + 1, dim=1, largest=False, sorted=True)[1].cpu().numpy()

            acc_dbscan, prec_dbscan, recall_dbscan, efficiency_dm_dbscan, efficiency_lhc_dbscan = calc_scores_with_efficiency_dbscan(
                K, k_list_dbscan, indices, dbscan_labels, batch_dbscan_labels)

            accuracy_scores_dbscan.extend(acc_dbscan)
            precision_scores_dbscan.extend(prec_dbscan)
            recall_scores_dbscan.extend(recall_dbscan)
            efficiency_dm_scores_dbscan.extend(efficiency_dm_dbscan)
            efficiency_lhc_scores_dbscan.extend(efficiency_lhc_dbscan)
            
        else:
            indices = dist_mat_batch.topk(K + 1, dim=1, largest=False, sorted=True)[1].cpu().numpy()
            acc_dbscan, prec_dbscan, recall_dbscan, efficiency_dm_dbscan, efficiency_lhc_dbscan = [], [], [], [], []
            acc_cluster, prec_cluster, recall_cluster, efficiency_dm_cluster, efficiency_lhc_cluster = calc_scores_with_efficiency_cluster_ids(K, k_list_cluster, indices, cluster_ids, batch_cluster_ids)
            accuracy_scores_cluster.extend(acc_cluster)
            precision_scores_cluster.extend(prec_cluster)
            recall_scores_cluster.extend(recall_cluster)
            efficiency_dm_scores_cluster.extend(efficiency_dm_cluster)
            efficiency_lhc_scores_cluster.extend(efficiency_lhc_cluster)

        

        

    # Return the average metrics for both clusters (if both are used)
    

    if use_dbscan:
        return np.mean(accuracy_scores_dbscan), np.mean(precision_scores_dbscan), np.mean(recall_scores_dbscan), np.mean(efficiency_dm_scores_dbscan), np.mean(efficiency_lhc_scores_dbscan)
    else:
        return np.mean(accuracy_scores_cluster), np.mean(precision_scores_cluster), np.mean(recall_scores_cluster), np.mean(efficiency_dm_scores_cluster),np.mean(efficiency_lhc_scores_cluster)
     


@jit(nopython=True)
def calc_scores_with_efficiency_cluster_ids(K, k_list, indices, cluster_ids, batch_cluster_ids):
    acc = []
    prec = []
    recall = []
    efficiency_dm = []  # To store efficiency scores
    efficiency_lhc = []  # To store efficiency scores

    for i, k in enumerate(k_list):
        if k == 0:
            continue

        neighbors = indices[i, 1 : K + 1]

        neighbor_labels = cluster_ids[neighbors]

        matches = neighbor_labels == batch_cluster_ids[i]

        accuracy = matches[:k].sum() / k
        precision_at_K = matches.sum() / K
        recall_at_K = matches.sum() / k

        true_neighbors = np.where(cluster_ids == batch_cluster_ids[i])[0]
        predicted_neighbors_set = set(neighbors)

        correctly_predicted_edges = matches.sum()  
        total_true_edges = len(true_neighbors) - 1

        if total_true_edges > 0:
            efficiency_ratio = correctly_predicted_edges / total_true_edges
            if efficiency_ratio <= 0:
                continue
            if efficiency_ratio > 0.5:
                efficiency_dm_at_K = efficiency_ratio
            if efficiency_ratio > 0.75:
                efficiency_lhc_at_K = efficiency_ratio
            
        else:
            continue  # Skip if there are no true edges
        acc.append(accuracy)
        prec.append(precision_at_K)
        recall.append(recall_at_K)
        efficiency_dm.append(efficiency_dm_at_K)
        efficiency_lhc.append(efficiency_lhc_at_K)

    return acc, prec, recall, efficiency_dm, efficiency_lhc

@jit(nopython=True)
def calc_scores_with_efficiency_dbscan(K, k_list, indices, dbscan_labels, batch_dbscan_labels):
    acc = []
    prec = []
    recall = []
    efficiency_dm = []  # To store efficiency scores
    efficiency_lhc = []  # To store efficiency scores

    for i, k in enumerate(k_list):
        if k == 0:
            continue

        neighbors = indices[i, 1:K + 1]

        neighbor_labels = dbscan_labels[neighbors]

        matches = neighbor_labels == batch_dbscan_labels[i]

        accuracy = matches[:k].sum() / k
        precision_at_K = matches.sum() / K
        recall_at_K = matches.sum() / k

        true_neighbors = np.where(dbscan_labels == batch_dbscan_labels[i])[0]
        predicted_neighbors_set = set(neighbors)

        correctly_predicted_edges = matches.sum()  
        total_true_edges = len(true_neighbors) - 1

        if total_true_edges > 0:
            efficiency_ratio = correctly_predicted_edges / total_true_edges
            if efficiency_ratio <= 0:
                continue
            if efficiency_ratio > 0.5:
                efficiency_dm_at_K = efficiency_ratio
            if efficiency_ratio > 0.75:
                efficiency_lhc_at_K = efficiency_ratio
            
        else:
            continue  # Skip if there are no true edges
        acc.append(accuracy)
        prec.append(precision_at_K)
        recall.append(recall_at_K)
        efficiency_dm.append(efficiency_dm_at_K)
        efficiency_lhc.append(efficiency_lhc_at_K)

    return acc, prec, recall, efficiency_dm, efficiency_lhc


def calculate_node_classification_metrics(pred, target, mask):
    pred = pred[mask]
    target = target[mask]
    acc = (pred == target).sum().item() / mask.sum().item()
    return acc
