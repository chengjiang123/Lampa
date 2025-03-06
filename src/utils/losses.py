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
    
    
    
@torch.compile
def condensation_loss_tiger(
    *,
    beta: T,
    x: T,
    point_pairs: T,
    cluster_ids: T,
    recons: T,
    pts: T,
    q_min: float,
    noise_threshold: int,
    max_n_rep: int,
) -> tuple[dict[str, T], dict[str, Union[int, float]]]:
    """Extracted function for torch compilation. See `condensation_loss_tiger` for
    docstring.

    Args:
        object_mask: Mask for the particles that should be considered for the loss
            this is broadcased to n_hits

    Returns:
        loss_dct: Dictionary of losses
        extra_dct: Dictionary of extra information
    """
    # _j means indexed by hits
    # _k means indexed by objects

    # To protect against nan in divisions
    eps = 1e-9

    cluster_ids = cluster_ids.cpu().numpy()
    unique_clusters, counts = np.unique(cluster_ids, return_counts=True)
    assert (
        len(unique_oids_k) > 0
    ), "No particles of interest found, cannot evaluate loss"
    # n_nodes x n_pids
    # The nodes in every column correspond to the hits of a single particle and
    # should attract each other
    
    all_pos_pair_mask = cluster_ids[point_pairs[0]] == cluster_ids[point_pairs[1]]

    extra_pos_pair_mask = pair_filter(cluster_ids, point_pairs, recons, pts, pt_thres=0.9)
    #all_pos_pair_mask = all_pos_pair_mask & extra_pos_pair_mask
    #all_neg_pair_mask = ~all_pos_pair_mask
    
    attractive_mask_jk = all_pos_pair_mask & extra_pos_pair_mask

    q_j = torch.arctanh(beta) ** 2 + q_min
    assert not torch.isnan(q_j).any(), "q contains NaNs"

    # Index of condensation points in node array
    alphas_k = torch.argmax(q_j.view(-1, 1) * attractive_mask_jk, dim=0)

    # 1 x n_objs
    q_k = q_j[alphas_k].view(1, -1)
    qw_jk = q_j.view(-1, 1) * q_k

    # n_objs x n_outdim
    x_k = x[alphas_k]
    dist_jk = torch.cdist(x, x_k)

    # Calculate normalization factors
    # -------------------------------
    n_hits = len(object_mask)
    # oi = of interest = not masked
    n_hits_oi = object_mask.sum()
    n_particles_oi = len(alphas_k)
    # every hit has a rep edge to every other CP except its own
    norm_rep = eps + (n_particles_oi - 1) * n_hits
    # need to subtract n_particle_oi to avoid double counting
    norm_att = eps + n_hits_oi - n_particles_oi

    # Attractive potential/loss
    # -------------------------
    qw_att = qw_jk[attractive_mask_jk]
    v_att = (qw_att * torch.square(dist_jk[attractive_mask_jk])).sum() / norm_att

    # Repulsive potential
    # -------------------
    repulsive_mask_jk = (~attractive_mask_jk) & (dist_jk < 1)
    n_rep = repulsive_mask_jk.sum()
    if n_rep > max_n_rep > 0:
        sampling_freq = max_n_rep / n_rep
        sampling_mask = (
            torch.rand_like(repulsive_mask_jk, dtype=torch.float16) < sampling_freq
        )
        repulsive_mask_jk &= sampling_mask
        norm_rep *= sampling_freq
    qw_rep_jk = qw_jk[repulsive_mask_jk]
    v_rep = (qw_rep_jk * (1 - dist_jk[repulsive_mask_jk])).sum() / norm_rep

    # Other losses
    # ------------
    l_coward = torch.mean(1 - beta[alphas_k])
    not_noise_j = object_id > noise_threshold
    l_noise = torch.mean(beta[~not_noise_j])

    loss_dct = {
        "attractive": v_att,
        "repulsive": v_rep,
        "coward": l_coward,
        "noise": l_noise,
    }
    extra_dct = {
        "n_rep": n_rep,
    }
    return loss_dct, extra_dct

class CondensationLossTiger(MultiLossFct, HyperparametersMixin):
    def __init__(
        self,
        *,
        lw_repulsive: float = 1.0,
        lw_noise: float = 0.0,
        lw_coward: float = 0.0,
        q_min: float = 0.01,
        pt_thld: float = 0.9,
        max_eta: float = 4.0,
        max_n_rep: int = 0,
        sample_pids: float = 1.0,
    ):
        """Implementation of condensation loss that directly calculates the n^2
        distance matrix.

        Args:
            lw_repulsive: Loss weight for repulsive part of potential loss
            lw_noise: Loss weight for noise loss
            lw_background: Loss weight for background loss
            q_min (float, optional): See OC paper. Defaults to 0.01.
            pt_thld (float, optional): pt thld for interesting particles. Defaults to 0.9.
            max_eta (float, optional): eta thld for interesting particles. Defaults to 4.0.
            max_n_rep (int, optional): Maximum number of repulsive edges to consider.
                Defaults to 0 (all).
            sample_pids (float, optional): Further subsample particles to conserve
                memory. Defaults to 1.0 (no sampling)
        """
        super().__init__()
        self.save_hyperparameters()

    # noinspection PyUnusedLocal
    def forward(
        self,
        *,
        beta: T,
        x: T,
        cluster_ids: T,
        reconstructable: T,
        pt: T,
        ec_hit_mask: None,
        eta: T,
        **kwargs,
    ) -> MultiLossFctReturn:
        if ec_hit_mask is not None:
            # If a post-EC node mask was applied in the model, then all model outputs
            # already include this mask, while everything gotten from the data
            # does not. Hence, we apply it here.
            particle_id = particle_id[ec_hit_mask]
            reconstructable = reconstructable[ec_hit_mask]
            pt = pt[ec_hit_mask]
            eta = eta[ec_hit_mask]
        
        losses, extra = condensation_loss_tiger(
            beta=beta,
            x=x,
            cluster_ids=cluster_ids,
            q_min=self.hparams.q_min,
            noise_threshold=0.0,
            max_n_rep=self.hparams.max_n_rep,
        )
        weights = {
            "attractive": 1.0,
            "repulsive": self.hparams.lw_repulsive,
            "noise": self.hparams.lw_noise,
            "coward": self.hparams.lw_coward,
        }
        return MultiLossFctReturn(
            loss_dct=losses,
            weight_dct=weights,
            extra_metrics=extra,
        )

