import argparse
import logging
import os
import pdb
import sys
import traceback
from typing import Optional

import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from data import DataFold
from data.binding_data_multitask import MultitaskTaskSampleBatchIterable
from models.abstract_torch_fsmol_model import (
    train_loop,
    create_optimizer,
)
from models.gnn_multitask import GNNMultitaskConfig, create_model
from modules.graph_feature_extractor import (
    add_graph_feature_extractor_arguments,
    make_graph_feature_extractor_config_from_args,
)
from utils.cli_utils import add_train_cli_args, set_up_train_run

SMALL_NUMBER = 1e-7
logger = logging.getLogger(__name__)

"""
To test on a small version of the ignite_dataset:
python ignite_src/multitask_train.py ignite_src/prototype_simulation_dataset --task-list-file ignite_src/datasets/ignite_data.json

To test on a large version of the ignite_dataset:
"""

def add_model_arguments(parser: argparse.ArgumentParser):
    add_graph_feature_extractor_arguments(parser)
    parser.add_argument("--num_tail_layers", type=int, default=2)


def make_model_from_args(
        num_tasks: int, args: argparse.Namespace, device: Optional[torch.device] = None
):
    model_config = GNNMultitaskConfig(
        graph_feature_extractor_config=make_graph_feature_extractor_config_from_args(args),
        num_tasks=num_tasks,
        num_tail_layers=args.num_tail_layers,
    )
    model = create_model(model_config, device=device)
    return model


def add_train_loop_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--cuda", type=int, default=5)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.00005,
        help="Learning rate for shared model components.",
    )
    parser.add_argument(
        "--metric-to-use",
        type=str,
        choices=[
            "acc",
            "balanced_acc",
            "f1",
            "prec",
            "recall",
            "roc_auc",
            "avg_precision",
            "kappa",
        ],
        default="avg_precision",
        help="Metric to evaluate on validation data.",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a Multitask GNN model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_train_cli_args(parser)
    add_model_arguments(parser)
    # Training parameters:
    add_train_loop_arguments(parser)
    parser.add_argument(
        "--task-specific-lr",
        type=float,
        default=0.0001,
        help="Learning rate for shared model components. By default, 10x core learning rate.",
    )
    parser.add_argument(
        "--finetune-lr-scale",
        type=float,
        default=1.0,
        help="Scaling factor for LRs used in finetuning eval.",
    )

    args = parser.parse_args()

    save_name = (f'{args.learning_rate}_{args.batch_size}_{args.num_heads}_{args.intermediate_dim}_'
                 f'{args.num_gnn_layers}_{args.readout_use_all_states}_{args.readout_head_dim}_{args.num_tail_layers}')
    out_dir, fsmol_dataset, aml_run = set_up_train_run(f"{save_name}_Multitask", args, torch=True)
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else "cpu")
    model = make_model_from_args(
        num_tasks=fsmol_dataset.get_num_fold_tasks(DataFold.TRAIN), args=args, device=device
    )
    logger.info(f"\tNum parameters {sum(p.numel() for p in model.parameters())}")
    logger.info(f"\tDevice: {device}")
    logger.info(f"\tModel:\n{model}")

    train_task_name_to_id = {
        name: i for i, name in enumerate(fsmol_dataset.get_task_names(data_fold=DataFold.TRAIN))
    }

    if args.task_specific_lr is not None:
        task_specific_lr = args.task_specific_lr
    else:
        task_specific_lr = 10 * args.learning_rate

    optimizer, lr_scheduler = create_optimizer(
        model,
        lr=args.learning_rate,
        task_specific_lr=task_specific_lr,
        warmup_steps=100,
        task_specific_warmup_steps=100,
    )

    # Validate on the held-out molecules.
    _, best_model_state = train_loop(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_data=MultitaskTaskSampleBatchIterable(
            fsmol_dataset,
            data_fold=DataFold.TRAIN,
            task_name_to_id=train_task_name_to_id,
            max_num_graphs=args.batch_size,
            device=device,
        ),
        valid_data=MultitaskTaskSampleBatchIterable(
            fsmol_dataset,
            data_fold=DataFold.VALIDATION,
            task_name_to_id=train_task_name_to_id,
            max_num_graphs=args.batch_size,
            device=device,
        ),
        max_num_epochs=args.num_epochs,
        patience=args.patience,
        aml_run=aml_run,
    )

    torch.save(best_model_state, os.path.join(out_dir, "best_model.pt"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
