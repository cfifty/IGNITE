from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from dpu_utils.utils import RichPath

import sys
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from data.fsmol_task import MoleculeDatapoint, get_task_name_from_path, GraphData

@dataclass(frozen=True)
class BindingAffinityTask:
    """Data structure to hold information from binding_affinities jsonl files.

    Args:
        name: String describing the task's name eg. "CHEMBL1000114".
        samples: List of MoleculeDatapoint samples associated with this task.
    """
    name: str
    samples: List[MoleculeDatapoint]

    @staticmethod
    def load_from_file(path: RichPath) -> "BindingAffinityTask":
        samples = []
        for raw_sample in path.read_by_file_suffix():
            graph_data = raw_sample.get("graph")
            adjacency_lists = []
            for adj_list in graph_data["adjacency_lists"]:
                if len(adj_list) > 0:
                    adjacency_lists.append(np.array(adj_list, dtype=np.int64))
                else:
                    adjacency_lists.append(np.zeros(shape=(0, 2), dtype=np.int64))
            samples.append(
                MoleculeDatapoint(
                    task_name=get_task_name_from_path(path),
                    smiles=raw_sample["SMILES"],
                    bool_label=False,
                    numeric_label=float(raw_sample.get("RegressionProperty") or "nan"),
                    fingerprint=None,
                    descriptors=None,
                    graph=GraphData(
                        node_features=np.array(graph_data["node_features"], dtype=np.float32),
                        adjacency_lists=adjacency_lists,
                        edge_features=[
                            np.array(edge_feats, dtype=np.float32)
                            for edge_feats in graph_data.get("edge_features") or []
                        ],
                    ),
                )
            )

        return BindingAffinityTask(get_task_name_from_path(path), samples)

@dataclass(frozen=True)
class BindingAffinityTaskSample:
    """Data structure output of a Task Sampler.

    Args:
        name: String describing the task's name eg. "CHEMBL1000114".
        train_samples: List of MoleculeDatapoint samples drawn as the support set.
        valid_samples: List of MoleculeDatapoint samples drawn as the validation set.
            This may be empty, dependent on the nature of the Task Sampler.
        test_samples: List of MoleculeDatapoint samples drawn as the query set.
    """

    name: str
    train_samples: List[MoleculeDatapoint]
    valid_samples: List[MoleculeDatapoint]
    test_samples: List[MoleculeDatapoint]

    @property
    def train_pos_label_ratio(self) -> float:
        raise ValueError('train_pos_label_ratio not defined for BindingAffinityTask')

    @property
    def test_pos_label_ratio(self) -> float:
        raise ValueError('test_pos_label_ratio not defined for BindingAffinityTask')
