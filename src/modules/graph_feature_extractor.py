import argparse
from dataclasses import dataclass
from typing import Optional
from typing_extensions import Literal

import torch
import torch.nn as nn

from data.fsmol_batcher import FSMolBatch
from data.fsmol_dataset import NUM_NODE_FEATURES
from modules.gnn import GNN, GNNConfig, add_gnn_model_arguments, make_gnn_config_from_args
from modules.graph_readout import (
    GraphReadoutConfig,
    add_graph_readout_arguments,
    make_graph_readout_config_from_args,
    make_readout_model,
)


@dataclass(frozen=True)
class GraphFeatureExtractorConfig:
    initial_node_feature_dim: int = NUM_NODE_FEATURES
    gnn_config: GNNConfig = GNNConfig()
    readout_config: GraphReadoutConfig = GraphReadoutConfig()
    output_norm: Literal["off", "layer", "batch"] = "off"


def add_graph_feature_extractor_arguments(parser: argparse.ArgumentParser):
    add_gnn_model_arguments(parser)
    add_graph_readout_arguments(parser)


def make_graph_feature_extractor_config_from_args(
        args: argparse.Namespace, initial_node_feature_dim: int = NUM_NODE_FEATURES
) -> GraphFeatureExtractorConfig:
    return GraphFeatureExtractorConfig(
        initial_node_feature_dim=initial_node_feature_dim,
        gnn_config=make_gnn_config_from_args(args),
        readout_config=make_graph_readout_config_from_args(args),
    )


class GraphFeatureExtractor(nn.Module):
    def __init__(self, config: GraphFeatureExtractorConfig):
        super().__init__()
        self.config = config

        # Initial (per-node) layers:
        self.init_node_proj = nn.Linear(
            config.initial_node_feature_dim, config.gnn_config.hidden_dim, bias=False
        )

        self.gnn = GNN(self.config.gnn_config)

        if config.readout_config.use_all_states:
            readout_node_dim = (config.gnn_config.num_layers + 1) * config.gnn_config.hidden_dim
        else:
            readout_node_dim = config.gnn_config.hidden_dim

        self.readout = make_readout_model(
            self.config.readout_config,
            readout_node_dim,
        )

        if self.config.output_norm == "off":
            self.final_norm_layer: Optional[torch.nn.Module] = None
        elif self.config.output_norm == "layer":
            self.final_norm_layer = nn.LayerNorm(
                normalized_shape=self.config.readout_config.output_dim
            )
        elif self.config.output_norm == "batch":
            self.final_norm_layer = nn.BatchNorm1d(
                num_features=self.config.readout_config.output_dim
            )

    def forward(self, input: FSMolBatch) -> torch.Tensor:
        # ----- Initial (per-node) layer:
        initial_node_features = self.init_node_proj(input.node_features)

        # ----- Message passing layers:
        all_node_representations = self.gnn(initial_node_features, input.adjacency_lists)

        # ----- Readout phase:
        if self.config.readout_config.use_all_states:
            readout_node_reprs = torch.cat(all_node_representations, dim=-1)
        else:
            readout_node_reprs = all_node_representations[-1]

        mol_representations = self.readout(
            node_embeddings=readout_node_reprs,
            node_to_graph_id=input.node_to_graph,
            num_graphs=input.num_graphs,
        )

        if self.final_norm_layer is not None:
            mol_representations = self.final_norm_layer(mol_representations)

        return mol_representations


class MATFeatureExtractor(GraphFeatureExtractor):

    def __init__(self, config: GraphFeatureExtractorConfig):
        super().__init__(config)
        self.initial_node_proj = nn.Identity()
        self.readout = make_readout_model(
            self.config.readout_config,
            128,  # Override this value because we're taking only the final layer node embeddings: not over all layers.
        )

    def forward(self, input: FSMolBatch) -> torch.Tensor:
        # ----- Message passing layers:
        all_node_representations = self.gnn(input)

        # Reshape from (b,num_nodes, embedding_dim) -> (b*num_nodes, embedding_dim)
        num_graphs = all_node_representations.shape[0]
        nodes_per_graph = all_node_representations.shape[1]
        all_node_representations = torch.reshape(all_node_representations, (num_graphs * nodes_per_graph, -1))
        node_to_graph = torch.repeat_interleave(
            torch.arange(nodes_per_graph, device=f'cuda:{all_node_representations.get_device()}'), num_graphs)

        mol_representations = self.readout(
            node_embeddings=all_node_representations,
            node_to_graph_id=node_to_graph,
            num_graphs=num_graphs,
        )

        if self.final_norm_layer is not None:
            mol_representations = self.final_norm_layer(mol_representations)

        return mol_representations
