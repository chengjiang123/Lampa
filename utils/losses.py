import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, segment_csr
from .metrics import pair_filter
from pytorch_lightning.core.mixins.hparams_mixin import HyperparametersMixin


import copy
from dataclasses import dataclass, field
from typing import Any,Union
from torch import Tensor as T

from scipy.optimize import linear_sum_assignment
import numpy as np


@dataclass
class MultiLossFctReturn:
    """Return type for loss functions that return multiple losses."""

    #: Split up losses
    loss_dct: dict[str, T]
    #: Weights
    weight_dct: Union[dict[str, T], dict[str, float]]
    #: Other things that should be logged
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.loss_dct.keys() == self.weight_dct.keys()

    @property
    def loss(self) -> T:
        loss = sum(self.weighted_losses.values())
        assert isinstance(loss, torch.Tensor)
        return loss

    @property
    def weighted_losses(self) -> dict[str, T]:
        return {k: v * self.weight_dct[k] for k, v in self.loss_dct.items()}
    
class MultiLossFct(torch.nn.Module):
    """Base class for loss functions that return multiple losses."""

    def forward(self, *args: Any, **kwargs: Any) -> MultiLossFctReturn: ...
        
        
        
class knnInfoNCELoss(nn.Module):
    def __init__(self, tau, dist_metric):
        super().__init__()
        self.tau = tau
        self.dist_metric = dist_metric

    def forward(self, x, edge_index, cluster_ids, recons, pts, pt_thres=0.9, **kwargs):
        device = x.device

        edge_index = edge_index.to(device)
        cluster_ids = cluster_ids.to(device)
        recons = recons.to(device)
        pts = pts.to(device)

        valid_mask = (recons != 0) & (pts > pt_thres)

        all_pos_pair_mask = cluster_ids[edge_index[0]] == cluster_ids[edge_index[1]]

        extra_pos_pair_mask = valid_mask[edge_index[0]] & valid_mask[edge_index[1]]
        all_pos_pair_mask = all_pos_pair_mask & extra_pos_pair_mask
  
        all_neg_pair_mask = ~all_pos_pair_mask

        similarity = self.calculate_similarity(x, edge_index)

        loss = self.calc_info_nce(x, similarity, edge_index, all_pos_pair_mask, all_neg_pair_mask)

        return loss

    def calculate_similarity(self, x, edge_index):
        if self.dist_metric == "cosine":
            similarity = F.cosine_similarity(x[edge_index[0]], x[edge_index[1]], dim=-1)
        elif self.dist_metric == "l2_rbf":
            l2_dist = torch.norm(x[edge_index[0]] - x[edge_index[1]], p=2, dim=-1)
            similarity = torch.exp(-l2_dist / (2 * 0.75**2))  # Assuming sigma = 0.75 for RBF kernel
        elif self.dist_metric == "l2":
            similarity = torch.norm(x[edge_index[0]] - x[edge_index[1]], p=2, dim=-1)
        else:
            raise NotImplementedError(f"Distance metric {self.dist_metric} not implemented.")
        return similarity

    def calc_info_nce(self, x, similarity, edge_index, pos_mask, neg_mask):
        device = x.device

        similarity = similarity.to(device)
        edge_index = edge_index.to(device)
        pos_mask = pos_mask.to(device)
        neg_mask = neg_mask.to(device)

        max_sim = (similarity / self.tau).max()
        exp_sim = torch.exp(similarity / self.tau - max_sim)

        pos_exp_sim = exp_sim[pos_mask]
        neg_exp_sim = exp_sim[neg_mask]

        numerator = pos_exp_sim

        group_indices = edge_index[0][neg_mask].to(device)

        denominator = deterministic_scatter(neg_exp_sim, group_indices, reduce="sum").clamp(min=0)

        denominator = denominator[edge_index[0][pos_mask].to(device)]

        loss_per_pos_pair = -torch.log(numerator / (numerator + denominator))

        return loss_per_pos_pair.mean()
    

    
class knnTripletLoss(nn.Module):
    def __init__(self, margin, dist_metric):
        super().__init__()
        self.margin = margin
        self.dist_metric = dist_metric

    def forward(self, x, edge_index, cluster_ids, recons, pts, pt_thres=0.9, **kwargs):
        device = x.device

        edge_index = edge_index.to(device)
        cluster_ids = cluster_ids.to(device)
        recons = recons.to(device)
        pts = pts.to(device)

        valid_mask = (recons != 0) & (pts > pt_thres)

        all_pos_pair_mask = cluster_ids[edge_index[0]] == cluster_ids[edge_index[1]]

        extra_pos_pair_mask = valid_mask[edge_index[0]] & valid_mask[edge_index[1]]
        all_pos_pair_mask = all_pos_pair_mask & extra_pos_pair_mask

        all_neg_pair_mask = ~all_pos_pair_mask

        similarity = self.calculate_similarity(x, edge_index)

        loss = self.calc_triplet(x, similarity, edge_index, all_pos_pair_mask, all_neg_pair_mask)

        return loss

    def calculate_similarity(self, x, edge_index):
        if self.dist_metric == "cosine":
            similarity = F.cosine_similarity(x[edge_index[0]], x[edge_index[1]], dim=-1)
        elif self.dist_metric == "l2_rbf":
            l2_dist = torch.norm(x[edge_index[0]] - x[edge_index[1]], p=2, dim=-1)
            similarity = torch.exp(-l2_dist / (2 * 0.75**2))  # Assuming sigma = 0.75 for RBF kernel
        elif self.dist_metric == "l2":
            similarity = torch.norm(x[edge_index[0]] - x[edge_index[1]], p=2, dim=-1)
        else:
            raise NotImplementedError(f"Distance metric {self.dist_metric} not implemented.")
        return similarity

    
    
    def calc_triplet(self, x, dists, edge_index, pos_mask, neg_mask):
        device = x.device

        dists = dists.to(device)
        edge_index = edge_index.to(device)
        pos_mask = pos_mask.to(device)
        neg_mask = neg_mask.to(device)

        pos_pair_dists = dists[pos_mask]

        group_indices = edge_index[0][neg_mask].to(device)

        neg_pair_dists = scatter_mean(dists[neg_mask], group_indices, dim=0).clamp(min=0)

        neg_pair_dists = neg_pair_dists[edge_index[0][pos_mask].to(device)]

        loss_per_pos_pair = torch.clamp(pos_pair_dists - neg_pair_dists + self.margin, min=0.0)

        # Return the average loss across all positive pairs
        return loss_per_pos_pair.mean()
    
class InfoNCELoss(nn.Module):
    def __init__(self, tau, dist_metric):
        super().__init__()
        self.tau = tau
        self.dist_metric = dist_metric

    def forward(self, x, point_pairs, cluster_ids, recons, pts, **kwargs):
        device = x.device
        
        point_pairs = point_pairs.to(device)
        cluster_ids = cluster_ids.to(device)
        
        all_pos_pair_mask = cluster_ids[point_pairs[0]] == cluster_ids[point_pairs[1]]

        extra_pos_pair_mask = pair_filter(cluster_ids, point_pairs, recons, pts, pt_thres=0.9)
        all_pos_pair_mask = all_pos_pair_mask & extra_pos_pair_mask
        all_neg_pair_mask = ~all_pos_pair_mask

        if self.dist_metric == "cosine":
            similarity = F.cosine_similarity(x[point_pairs[0]], x[point_pairs[1]], dim=-1)
        elif self.dist_metric == "l2_rbf":
            l2_dist = torch.linalg.norm(x[point_pairs[0]] - x[point_pairs[1]], ord=2, dim=-1)
            # l2_dist = batched_point_distance(x, point_pairs, batch_size=5000)
            sigma = 0.75
            similarity = torch.exp(-l2_dist / (2 * sigma**2))
        elif self.dist_metric == "l2_inverse":
            l2_dist = torch.linalg.norm(x[point_pairs[0]] - x[point_pairs[1]], ord=2, dim=-1)
            similarity = 1.0 / (l2_dist + 1.0)
        else:
            raise NotImplementedError

        loss_per_pos_pair = self.calc_info_nce(x, similarity, point_pairs, all_pos_pair_mask, all_neg_pair_mask)
        new_labels = cluster_ids[point_pairs[0][all_pos_pair_mask]]  # [topk_mask]
        unique_new_labels, new_labels = torch.unique(new_labels, return_inverse=True)
        loss_per_pos_pair = deterministic_scatter(loss_per_pos_pair, new_labels, reduce="mean")

        return torch.mean(loss_per_pos_pair)

    def calc_info_nce(self, x, similarity, all_pairs, all_pos_pair_mask, all_neg_pair_mask):
        device = x.device
        
        similarity = similarity.to(device)
        all_pairs = all_pairs.to(device)
        all_pos_pair_mask = all_pos_pair_mask.to(device)
        all_neg_pair_mask = all_neg_pair_mask.to(device)
        
        max_sim = (similarity / self.tau).max()
        exp_sim = torch.exp(similarity / self.tau - max_sim)

        pos_exp_sim = exp_sim[all_pos_pair_mask]
        neg_exp_sim = exp_sim[all_neg_pair_mask]

        numerator = pos_exp_sim
        group_indices = all_pairs[0][all_neg_pair_mask].to(device)
        denominator = deterministic_scatter(neg_exp_sim, group_indices, reduce="sum").clamp(min=0)

        denominator = denominator[all_pairs[0][all_pos_pair_mask].to(device)]
        loss_per_pos_pair = -torch.log(numerator / (numerator + denominator))
        return loss_per_pos_pair

    def calc_triplet(self, x, dists, all_pairs, all_pos_pair_mask, all_neg_pair_mask):
        group_indices = all_pairs[0][all_neg_pair_mask]
        neg_pair_dists = torch.zeros(x.shape[0], device=x.device)
        neg_pair_dists = scatter_mean(dists[all_neg_pair_mask], group_indices, out=neg_pair_dists)
        neg_pair_dists = neg_pair_dists[all_pairs[0][all_pos_pair_mask]]

        loss_per_pos_pair = torch.clamp(dists[all_pos_pair_mask] - neg_pair_dists + self.margin, min=0.0)
        return loss_per_pos_pair


def deterministic_scatter(src, index, reduce):
    sorted_arg = torch.argsort(index)
    sorted_index = index[sorted_arg]
    sorted_src = src[sorted_arg]
    unique_groups, counts = torch.unique_consecutive(sorted_index, return_counts=True)
    indptr = torch.zeros(len(unique_groups) + 1, device=src.device)
    indptr[1:] = torch.cumsum(counts, dim=0)
    output = segment_csr(sorted_src, indptr.long(), reduce=reduce)
    return output


def batched_point_distance(x, point_pairs, batch_size=1000):
    """
    Compute the L2 norm between points in x specified by point_pairs in batches.

    :param x: Tensor of shape (n, d)
    :param point_pairs: Tensor of shape (2, E)
    :param batch_size: Size of the batch for processing
    :return: Tensor of distances
    """
    num_pairs = point_pairs.size(1)
    distances = []

    for i in range(0, num_pairs, batch_size):
        batch_pairs = point_pairs[:, i : i + batch_size]
        diff = x[batch_pairs[0]] - x[batch_pairs[1]]
        batch_distances = torch.linalg.norm(diff, ord=2, dim=-1)
        distances.append(batch_distances)

    return torch.cat(distances)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
    
    
    


    
    
# === Matching Cost Functions ===

def dice_loss_cost(inputs, targets, num_boxes,mask=None):
    # Flatten spatial dimensions (assume input shape [*, H, W] or [*, N])
    pred = inputs.sigmoid().flatten(1)     # [Q, N]
    tgt = targets.flatten(1)                # [T, N]
    if mask is not None:
        pred = pred * mask.flatten(1)
        tgt = tgt * mask.flatten(1)
    # Expand dimensions for pairwise computation
    pred = pred.unsqueeze(1)   # [Q, 1, N]
    tgt = tgt.unsqueeze(0)     # [1, T, N]
    # Compute dice loss per pair
    intersection = 2 * (pred * tgt).sum(dim=2)         # [Q, T]
    denominator = pred.sum(dim=2) + tgt.sum(dim=2)       # [Q, T]
    loss = 1 - (intersection + 1) / (denominator + 1)    # [Q, T]
    #print(loss)
    return loss  # Returns a [Q, T] cost matrix


def sigmoid_bce_cost(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: (num_querys, N)
        targets: (num_inst, N)
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))

    return loss / N


def compute_dice_loss(inputs, targets, num_boxes,mask=None):
    # Flatten spatial dimensions (assume input shape [*, H, W] or [*, N])
    pred = inputs.sigmoid().flatten(1)     # [Q, N]
    tgt = targets.flatten(1)                # [T, N]
    if mask is not None:
        pred = pred * mask.flatten(1)
        tgt = tgt * mask.flatten(1)
    # Expand dimensions for pairwise computation
    pred = pred.unsqueeze(1)   # [Q, 1, N]
    tgt = tgt.unsqueeze(0)     # [1, T, N]
    # Compute dice loss per pair
    intersection = 2 * (pred * tgt).sum(dim=2)         # [Q, T]
    denominator = pred.sum(dim=2) + tgt.sum(dim=2)       # [Q, T]
    loss = 1 - (intersection + 1) / (denominator + 1)    # [Q, T]
    return loss.sum() / (num_boxes + 1e-6)  # Returns a scalar


def compute_sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, mask=None):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    if mask is not None:
        ce_loss = ce_loss * mask
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / (num_boxes + 1e-6)

# === Hungarian Matcher with Flat Index Output ===

class HungarianMatcher(nn.Module):
    """
    Hungarian matcher for MaskFormer when class labels are not available.
    It matches predictions to targets based solely on mask cost (dice and BCE).
    Additionally, it ensures a unique, consistent indexing of predictions
    across all loss components using a custom lap function.
    """
    def __init__(self, ignore_label=-100, cost_mask=5, cost_dice=5, default_idx=None, n_queries=100):
        """
        Args:
            ignore_label:   Label value to ignore in the ground truth.
            cost_mask:      Relative weight for the mask BCE loss.
            cost_dice:      Relative weight for the mask dice loss.
            default_idx:    Set of all possible prediction indices (if None, defaults to 0-99).
        """
        super().__init__()
        self.ignore_label = ignore_label
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        # Ensure at least one cost is non-zero.
        assert cost_mask != 0 or cost_dice != 0, "All costs cannot be 0"
        # Set default indices for unmatched predictions.
        #self.default_idx = default_idx if default_idx is not None else set(range(100))
        self.default_idx = set(range(n_queries))
        self.global_step = 0

    def get_gt(self, instance_labels_b):
        """
        Process ground truth instance masks from instance labels.
        Args:
            instance_labels_b (Tensor): A 1D tensor of length N, where each element
                                        indicates the instance ID for that point.
        Returns:
            mask_labels_b (Tensor): A tensor of shape [num_instances, N] where each row
                                    is a binary mask for a unique instance.
        """
        unique_inst = torch.unique(instance_labels_b)
        unique_inst = unique_inst[unique_inst != self.ignore_label]
        n_inst_gt = len(unique_inst)
        if n_inst_gt == 0:
            return None
        n_points = instance_labels_b.shape[0]
        mask_labels_b = torch.zeros((n_inst_gt, n_points), 
                                    device=instance_labels_b.device, 
                                    dtype=torch.float)
        for i, inst_id in enumerate(unique_inst):
            mask_labels_b[i] = (instance_labels_b == inst_id).float()
        return mask_labels_b

    def get_match(self, gt_masks, pred_masks):
        """
        Computes a cost matrix between each predicted mask and each ground-truth mask.
        Args:
            gt_masks (Tensor): [num_targets, N] binary masks.
            pred_masks (Tensor): [n_queries, N] predicted mask logits.
        Returns:
            cost (Tensor): Cost matrix of shape [n_queries, num_targets].
        """
        dice_cost = dice_loss_cost(pred_masks, gt_masks, num_boxes=pred_masks.shape[0])

        #bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks, reduction="none").mean(1)
        #bce_cost = bce_loss.unsqueeze(1).repeat(1, gt_masks.shape[0])
        bce_cost = sigmoid_bce_cost(pred_masks, gt_masks)
        C = self.cost_dice * dice_cost + self.cost_mask * bce_cost
        #C = self.cost_mask * bce_cost
        return C


    def lap(self, cost):
        """
        Performs Hungarian matching and ensures that the resulting indices
        cover all predictions by appending default indices for unmatched queries.
        Args:
            cost (ndarray): Cost matrix.
        Returns:
            List[int]: A list of prediction indices of length equal to the number of queries.
        """
        src_idx, _ = linear_sum_assignment(cost)
        assigned = set(src_idx)
        remaining = sorted(self.default_idx - assigned)
        # Concatenate assigned indices with defaults to ensure full coverage.
        return list(src_idx) + remaining

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Should contain key 'pred_masks' of shape [1, n_queries, H*W].
            targets (list): A list (length=1) of target dictionaries.
                            Each target dict must contain key 'instances' (Tensor of shape [H*W]).
        Returns:
            idx (Tensor): 1D tensor of shape [n_queries] with the matched ground-truth mask index for each query.
        """
        # Expecting single batch.
        n_queries = outputs["pred_masks"].shape[1]
        #print(f'targets[0]["instances"] {targets[0]["instances"]}')
        gt_masks = self.get_gt(targets[0]["instances"])
        if gt_masks is None:
            # If no ground-truth exists, assign a dummy mapping.
            dummy_cost = torch.full((n_queries, n_queries), 1e8)
            row_ind, _ = linear_sum_assignment(dummy_cost.cpu().numpy())
            idx = row_ind.tolist()
        else:
            out_masks = outputs["pred_masks"][0]  # [n_queries, H*W]
            C = self.get_match(gt_masks, out_masks)
            C[C.isnan() | C.isinf()] = 1e8  # safeguard
            row_ind, col_ind = linear_sum_assignment(C.detach().cpu().numpy())
            # Here we assume that every query gets assigned (or, if not, adjust as needed).
            # We build a full mapping for all queries.
            idx = [-1] * n_queries
            for r, c in zip(row_ind, col_ind):
                idx[r] = c
            # For queries not assigned a ground-truth (if any), assign a default index (e.g. 0).
            valid = [i for i in range(n_queries) if idx[i] != -1]
        idx = torch.tensor(idx, device=outputs["pred_masks"].device, dtype=torch.long)
        self.global_step += 1
        return idx

# ---------------- Criterion (Updated for Single Batch Matcher) ----------------
class SegCriterion(nn.Module):
    """
    Segmentation loss criterion that uses the single-batch HungarianMatcher.
    It matches the predicted masks with the ground-truth binary masks (from get_gt)
    and computes a combination of dice loss, BCE loss, and IoU loss.
    """
    def __init__(self, tau, dist_metric, cost_dice=1.0, cost_bce=1.0, ignore_label=-100,n_queries=100):
        super().__init__()
        self.matcher = HungarianMatcher(ignore_label=ignore_label, cost_mask=cost_bce, cost_dice=cost_dice,n_queries=n_queries)
        self.ignore_label = ignore_label
        self.dist_metric = dist_metric
        self.tau = tau
        self.knn_loss = knnInfoNCELoss(tau=tau, dist_metric=dist_metric)

    def forward(self, outputs, targets, x, edge_index, cluster_ids, recons, pts, pt_thres=0.9, **kwargs):
        """
        Args:
            outputs (dict): Contains 'pred_masks' of shape [1, n_queries, H*W].
            targets (list): A list (length=1) of target dictionaries with key 'instances'.
        Returns:
            total_loss (Tensor): Combined segmentation loss.
        """
        pred_masks = outputs["pred_masks"].squeeze(0)  # [n_queries, H*W]
        n_queries = pred_masks.shape[0]
        idx = self.matcher(outputs, targets)           # [n_queries]
        print(f'idx {idx}')
         # Get GT instance masks (shape [K, N])
        gt_masks = self.matcher.get_gt(targets[0]["instances"])

        if gt_masks is None:
            return torch.tensor(0.0, device=pred_masks.device, requires_grad=True)

        # === Safely align matched GT masks ===
        matched_gt = torch.zeros_like(pred_masks)
        for i in range(n_queries):
            gt_idx = idx[i].item()
            if gt_idx >= 0 and gt_idx < gt_masks.shape[0]:
                matched_gt[i] = gt_masks[gt_idx]  # valid match
            else:
                matched_gt[i] = 0.0  # unmatched query = background
        matched_gt = gt_masks[idx]  # [n_queries, H*W]
        dice_loss = compute_dice_loss(pred_masks, matched_gt, num_boxes=pred_masks.shape[0])
        bce_loss = F.binary_cross_entropy_with_logits(pred_masks, matched_gt, reduction="none").mean(1).sum() / (pred_masks.shape[0] + 1e-6)
        iou = get_iou(pred_masks, matched_gt)
        iou_loss = F.mse_loss(iou, torch.ones_like(iou), reduction="sum") / (pred_masks.shape[0] + 1e-6)
        dice_loss = 0.5 * dice_loss 
        bce_loss = 3.0 * bce_loss
        iou_loss = 1.0 * iou_loss
        #infonce_loss = self.knn_loss(x, edge_index, cluster_ids, recons, pts, pt_thres=0.9, **kwargs)
        print(f'dice loss {dice_loss}')
        print(f'bce loss {bce_loss}')
        print(f'iou loss {iou_loss}')
        #print(f'infonce loss {infonce_loss}')
        total_loss = dice_loss + bce_loss + iou_loss
        #total_loss = infonce_loss
        return total_loss    
    
    
@torch.no_grad()
def get_iou(inputs, targets, thresh=0.5):
    inputs_bool = inputs.detach().sigmoid()
    inputs_bool = inputs_bool >= thresh

    intersection = (inputs_bool * targets).sum(-1)
    union = inputs_bool.sum(-1) + targets.sum(-1) - intersection

    iou = intersection / (union + 1e-6)

    return iou