import math

import torch
import torch.nn as nn
from pytorch_lightning.core.mixins.hparams_mixin import HyperparametersMixin
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask

from dynamic_edge_conv import DynamicEdgeConv
from edge_classifier import ECForGraphTCN
from interaction_network import InteractionNetwork as IN
from mlp import MLP, HeterogeneousResFCNN, ResFCNN
from resin import ResIN
from lightning import obj_from_or_to_hparams
from typing import Union


class INConvBlock(nn.Module):
    def __init__(
        self,
        indim,
        h_dim,
        e_dim,
        L,
        k,
        hidden_dim=100,
    ):
        super().__init__()
        self.relu = nn.ReLU()
        self.node_encoder = MLP(2 * indim, h_dim, hidden_dim=hidden_dim, L=1)
        self.edge_conv = DynamicEdgeConv(self.node_encoder, aggr="add", k=k)
        self.edge_encoder = MLP(2 * h_dim, e_dim, hidden_dim=hidden_dim, L=1)
        layers = []
        for _ in range(L):
            layers.append(
                IN(
                    node_indim=h_dim,
                    edge_indim=e_dim,
                    node_outdim=h_dim,
                    edge_outdim=e_dim,
                    node_hidden_dim=hidden_dim,
                    edge_hidden_dim=hidden_dim,
                )
            )
        self.layers = nn.ModuleList(layers)
        

    def forward(
        self,
        x: Tensor,
        alpha: float = 0.5,
    ) -> Tensor:
        h, edge_index = self.edge_conv(x)
        h = self.relu(h)
        edge_attr = torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=1)
        edge_attr = self.relu(self.edge_encoder(edge_attr))

        # apply the track condenser
        for layer in self.layers:
            delta_h, edge_attr = layer(h, edge_index, edge_attr)
            h = alpha * h + (1 - alpha) * delta_h
        return h
    
    
    
class ModularGraphTCN(nn.Module, HyperparametersMixin):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        ec: Union[nn.Module, None] = None,
        hc_in: nn.Module,
        node_indim: int,
        edge_indim: int,
        h_dim: int = 5,
        e_dim: int = 4,
        h_outdim: int = 2,
        hidden_dim: int = 40,
        feed_edge_weights: bool = False,
        ec_threshold: float = 0.5,
        mask_orphan_nodes: bool = False,
        use_ec_embeddings_for_hc: bool = False,
        alpha_latent: float = 0.0,
        n_embedding_coords: int = 0,
        heterogeneous_node_encoder: bool = False,
    ):
        """Track condensation network based on preconstructed graphs. This module
        combines the following:

        * Node and edge encoders to get to `(h_dim, e_dim)`
        * a track condensation network `hc_in`
        * an optional edge classifier

        Additional options configure how output from the edge classifier can be included
        in the track condensation network.

        Args:
            ec: Edge classifier
            hc_in: Track condensor interaction network.
            node_indim: Node feature dimension
            edge_indim: Edge feature dimension
            h_dim: node dimension in the condensation interaction networks
            e_dim: edge dimension in the condensation interaction networks
            h_outdim: output dimension in clustering space
            hidden_dim: width of hidden layers in all perceptrons
            feed_edge_weights: whether to feed edge weights to the track condenser
            ec_threshold: threshold for edge classification
            mask_orphan_nodes: Mask nodes with no connections after EC
            use_ec_embeddings_for_hc: Use edge classifier embeddings as input to
                track condenser. This currently assumes that h_dim and e_dim are
                also the dimensions used in the EC.
            alpha_latent: Assume that we're already starting from a latent space given
                by the first ``h_outdim`` node features. In this case, this is the
                strength of the residual connection
            n_embedding_coords: Number of embedding coordinates for which to add a
                residual connection. To be used with `alpha_latent`.
            heterogeneous_node_encoder: Whether to use different encoders for pixel/strip
        """
        
        
        super().__init__()
        self.save_hyperparameters(ignore=["ec", "hc_in"])
        self.relu = nn.ReLU()

        #: Edge classification network
        self.ec = obj_from_or_to_hparams(self, "ec", ec)
        #: Track condensation network (usually made up of interaction networks)
        self.hc_in = obj_from_or_to_hparams(self, "hc_in", hc_in)

        node_enc_indim = node_indim
        edge_enc_indim = edge_indim
        if use_ec_embeddings_for_hc:
            ec_node_latent_dim, ec_edge_latent_dim = ec.latent_dim
            node_enc_indim += int(ec_node_latent_dim)
            edge_enc_indim += int(ec_edge_latent_dim)
        edge_enc_indim += int(feed_edge_weights)

        #: Edge encoder network for track condenser
        self.hc_edge_encoder = MLP(
            edge_enc_indim,
            e_dim,
            hidden_dim=hidden_dim,
            L=2,
            bias=False,
        )
        
        # The fact that we use both MLP and ResFCNN is more historically
        if not heterogeneous_node_encoder:
            #: Node encoder network for track condenser
            self.hc_node_encoder = ResFCNN(
                in_dim=node_enc_indim,
                out_dim=h_dim,
                hidden_dim=hidden_dim,
                # depth = 1 for backwards compat, note that this is
                # equivalent to L=2
                depth=1,
                bias=False,
                alpha=0,
            )
        else:
            self.hc_node_encoder = HeterogeneousResFCNN(
                in_dim=node_enc_indim,
                out_dim=h_dim,
                hidden_dim=hidden_dim,
                depth=2,
                bias=False,
                alpha=0,
            )

        #: NN to predict beta
        self.p_beta = MLP(h_dim, 1, hidden_dim, L=3)
        #: NN to predict cluster coordinates
        self.p_cluster = MLP(h_dim, h_outdim, hidden_dim, L=3)
        #: NN to predict track parameters
        # self.p_track_param = IN(
        #     node_indim=h_dim,
        #     edge_indim=e_dim + hc_in.length_concatenated_edge_attrs,
        #     node_outdim=1,
        #     edge_outdim=1,
        #     node_hidden_dim=hidden_dim,
        #     edge_hidden_dim=hidden_dim,
        # )
        self._latent_normalization = torch.nn.Parameter(
            torch.Tensor([1.0]), requires_grad=True
        )
        
       
    def forward(
        self,
        data: Data,
    ) -> dict[str, Union[Tensor, None]]:
        edge_weights_unmasked = None
        edge_mask = None
        hit_mask = None
        if self.ec is not None:
            ec_result = self.ec(data)
            # Assign all EC  output to the data object, so that the cuts
            # will be applied automatically when we call `data.subgraph(...)` etc.
            data.edge_weights = ec_result["W"].reshape((-1, 1))
            data.ec_node_embedding = ec_result.get("node_embedding", None)
            data.ec_edge_embedding = ec_result.get("edge_embedding", None)
            edge_weights_unmasked = data.edge_weights.squeeze()
            edge_mask = (data.edge_weights > self.hparams.ec_threshold).squeeze()
            data = data.edge_subgraph(edge_mask)

            if self.hparams.mask_orphan_nodes:
                # Edge features do not need to be updated since there
                # are no loops (not affected by labeling)
                connected_nodes = data.edge_index.flatten().unique()
                hit_mask = index_to_mask(connected_nodes, size=data.num_nodes)
                data = data.subgraph(connected_nodes)
            else:
                hit_mask = torch.ones(
                    data.num_nodes, dtype=torch.bool, device=data.x.device
                )
        if self.ec is None and self.hparams.feed_edge_weights:
            data.edge_weights = data.ec_score.reshape((-1, 1))
            
            
 
        # Get the encoded inputs for the track condenser
        _edge_attrs = [data.edge_attr]
        _xs = [data.x]
        if self.hparams.use_ec_embeddings_for_hc:
            assert data.ec_edge_embedding is not None
            assert data.ec_node_embedding is not None
            _edge_attrs.append(data.ec_edge_embedding)
            _xs.append(data.ec_node_embedding)
        if self.hparams.feed_edge_weights:
            _edge_attrs.append(data.edge_weights)
        x = torch.cat(_xs, dim=1)
        edge_attrs = torch.cat(_edge_attrs, dim=1)
        h_hc = self.relu(self.hc_node_encoder(x, layer=data.layer))
        edge_attr_hc = self.relu(self.hc_edge_encoder(edge_attrs))

        # Run the track condenser
        h_hc, _, _ = self.hc_in(h_hc, data.edge_index, edge_attr_hc)
        beta = torch.sigmoid(self.p_beta(h_hc))
        # Soft clipping to protect against nans when calling arctanh(beta)
        assert not torch.isnan(beta).any()
        epsilon = 1e-6
        beta = epsilon + (1 - 2 * epsilon) * beta
        
        
        h = self.p_cluster(h_hc)
        if alpha_residue := self.hparams.alpha_latent:
            nec: int = self.hparams.n_embedding_coords
            assert nec > 0
            assert nec <= h.shape[1]
            _pad = (0, h.shape[1] - nec)
            residual = nn.functional.pad(data.x[:, :nec], _pad)
            h = math.sqrt(alpha_residue) * residual + math.sqrt(1 - alpha_residue) * h
        h *= self._latent_normalization
        # track_params, _ = self.p_track_param(
        #     h_hc, data.edge_index, torch.cat(edge_attrs_hc, dim=1)
        # )
        return {
            "W": edge_weights_unmasked,
            "H": h,
            "B": beta.squeeze(),
            "ec_hit_mask": hit_mask,
            "ec_edge_mask": edge_mask,
        }
    
    
    
class GraphTCN(nn.Module, HyperparametersMixin):
    def __init__(
        self,
        node_indim: int,
        edge_indim: int,
        *,
        h_dim=5,
        e_dim=4,
        h_outdim=2,
        hidden_dim=40,
        L_ec=3,
        L_hc=3,
        alpha_ec: float = 0.5,
        alpha_hc: float = 0.5,
        **kwargs,
    ):
        """`ModularTCN` with `ECForGraphTCN` as
        edge classification step and several interaction networks as residual layers
        for the track condensor network.

        This is a small wrapper around `ModularGraphTCN`, mostly to make sure that
        we can change the underlying implementation without invalidating config
        files that reference this class.

        Args:
            node_indim: Node feature dim
            edge_indim: Edge feature dim
            h_dim: node dimension in latent space
            e_dim: edge dimension in latent space
            h_outdim: output dimension in clustering space
            hidden_dim: width of hidden layers in all perceptrons
            L_ec: message passing depth for edge classifier
            L_hc: message passing depth for track condenser
            alpha_ec: strength of residual connection for multi-layer interaction
                networks in edge classifier
            alpha_hc: strength of residual connection for multi-layer interaction
                networks in track condenser
            **kwargs: Additional keyword arguments passed to `ModularGraphTCN`
        """
        
        
        super().__init__()
        self.save_hyperparameters()
        ec = ECForGraphTCN(
            node_indim=node_indim,
            edge_indim=edge_indim,
            hidden_dim=hidden_dim,
            interaction_node_dim=h_dim,
            interaction_edge_dim=e_dim,
            L_ec=L_ec,
            alpha=alpha_ec,
        )
        # Todo: Add other resin options
        hc_in = ResIN(
            node_dim=h_dim,
            edge_dim=e_dim,
            object_hidden_dim=hidden_dim,
            relational_hidden_dim=hidden_dim,
            alpha=alpha_hc,
            n_layers=L_hc,
        )
        self._gtcn = ModularGraphTCN(
            ec=ec,
            hc_in=hc_in,
            node_indim=node_indim,
            edge_indim=edge_indim,
            h_dim=h_dim,
            e_dim=e_dim,
            h_outdim=h_outdim,
            hidden_dim=hidden_dim,
            **kwargs,
        )
        
        

    def forward(
        self,
        data: Data,
    ) -> dict[str, Union[Tensor, None]]:
        return self._gtcn.forward(data=data)
    
    
        
        

        
        
        
        