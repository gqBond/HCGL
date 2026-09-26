"""Train continual heterogeneous graph-learning methods."""

import argparse
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from core.data_loader import load_data
from core.task_data import get_task_data
from core.training_utils import TrainingManager, get_ids_per_cls
from model.model_hetero import HAN, HAN_freebase
from model.model_hetero_multi import HAN_IMDB
from utils import EarlyStopping, set_random_seed, str2dict, str_to_bool


METHODS = [
    "finetune",
    "ewc",
    "mas",
    "twp",
    "ergnn",
    "dmsg",
    "ftf_er",
    "metacl_graph",
    "hero",
]


def build_model(dataset_name, meta_paths, input_size, output_size, args):
    model_args = {
        "meta_paths": meta_paths,
        "in_size": input_size,
        "hidden_size": args["hidden_units"],
        "out_size": output_size,
        "num_heads": args["num_heads"],
        "dropout": args["dropout"],
    }
    if dataset_name == "Freebase":
        return HAN_freebase(**model_args)
    if dataset_name == "IMDB":
        return HAN_IMDB(**model_args)
    if dataset_name == "DBLP":
        return HAN(**model_args)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def train(args):
    set_random_seed(args["seed"])
    os.environ["CGL_DATA_ROOT"] = str(Path(args["data_root"]).resolve())
    if args["method"] == "metacl_graph":
        args["patience"] = 200
    if args["dataset"] == "DBLP":
        args["cls_balance"] = True

    device = torch.device(args["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device} was requested, but CUDA is not available. "
            "Use --device cpu or install a CUDA-enabled PyTorch build."
        )

    output_dir = Path(args["output_dir"]).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Dataset: {args['dataset']}")
    print(f"Method: {args['method']}")
    print(f"Tasks: {args['num_tasks']}")
    print(f"Seed: {args['seed']}")
    print(f"Device: {device}")
    print(f"Patience: {args['patience']}")
    print(f"Class-balanced metric: {args['cls_balance']}")
    print("=" * 60)

    dataset = load_data(
        args["dataset"],
        feat_type=0,
        use_cache=args["use_cache"],
    )
    (
        full_graph,
        full_features,
        full_labels,
        num_classes,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        meta_paths,
    ) = dataset

    if num_classes % args["num_tasks"] != 0:
        raise ValueError(
            f"{num_classes} classes cannot be split evenly across "
            f"{args['num_tasks']} tasks."
        )

    classes_per_task = num_classes // args["num_tasks"]
    model = build_model(
        args["dataset"],
        meta_paths,
        full_features.shape[1],
        classes_per_task,
        args,
    ).to(device)

    training_full_graph = full_graph
    training_full_features = full_features
    training_full_labels = full_labels
    if args["method"] in {"dmsg", "ftf_er", "metacl_graph"}:
        training_full_graph = full_graph.to(device)
        training_full_features = full_features.to(device)
        training_full_labels = full_labels.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    method_module = importlib.import_module(f"life_model.{args['method']}")
    if args["method"] == "ftf_er":
        Path(args["data_path"]).mkdir(parents=True, exist_ok=True)
    if args["method"] in {"dmsg", "ftf_er"}:
        life_model = method_module.NET(model, training_full_graph, args)
    else:
        life_model = method_module.NET(model, args)

    manager = TrainingManager(model, life_model, loss_function, args)
    accuracy_matrix = np.zeros((args["num_tasks"], args["num_tasks"]))
    training_seconds = []
    inference_seconds = []

    for task_id in range(args["num_tasks"]):
        print(f"\nTask {task_id + 1}/{args['num_tasks']}")
        graph, features, labels, train_ids, val_ids, _ = get_task_data(
            dataset,
            args,
            task_id,
            use_cache=args["use_cache"],
        )
        graph = graph.to(device)
        features = features.to(device)
        labels = labels.to(device)
        train_ids = train_ids.to(device)
        val_ids = val_ids.to(device)

        train_ids_per_class = get_ids_per_cls(train_ids, labels)
        val_ids_per_class = get_ids_per_cls(val_ids, labels)
        checkpoint_path = checkpoint_dir / (
            f"{args['dataset']}_{args['method']}_seed{args['seed']}_"
            f"task{task_id}.pt"
        )
        stopper = EarlyStopping(
            patience=args["patience"],
            checkpoint_path=checkpoint_path,
        )
        start = time.perf_counter()

        for epoch in range(args["num_epochs"]):
            epoch_start = time.perf_counter()
            loss = manager.train_epoch(
                graph,
                features,
                labels,
                task_id,
                train_ids,
                ids_per_cls_train=train_ids_per_class,
                dataset=dataset,
                hg=training_full_graph,
                features_full=training_full_features,
                labels_full=training_full_labels,
            )

            val_loss, val_score = manager.validate(
                graph,
                features,
                labels,
                val_ids,
                val_ids_per_class,
            )
            should_stop = stopper.step(val_loss, val_score, model)

            if (epoch + 1) % args["log_every"] == 0 or epoch == 0:
                elapsed = time.perf_counter() - epoch_start
                print(
                    f"Epoch {epoch + 1:03d} | loss {loss.item():.4f} | "
                    f"val {val_score:.4f} | {elapsed:.3f}s"
                )
            if should_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        training_seconds.append(time.perf_counter() - start)
        stopper.load_checkpoint(model, map_location=device)

        start = time.perf_counter()
        for test_task_id in range(task_id + 1):
            test_graph, test_features, test_labels, _, _, test_ids = get_task_data(
                dataset,
                args,
                test_task_id,
                use_cache=args["use_cache"],
            )
            test_graph = test_graph.to(device)
            test_features = test_features.to(device)
            test_labels = test_labels.to(device)
            test_ids = test_ids.to(device)
            test_ids_per_class = get_ids_per_cls(test_ids, test_labels)
            test_loss, test_score = manager.test(
                test_graph,
                test_features,
                test_labels,
                test_ids,
                test_ids_per_class,
            )
            accuracy_matrix[task_id, test_task_id] = round(test_score * 100, 2)
            print(
                f"Test task {test_task_id + 1}: "
                f"loss={test_loss:.4f}, score={test_score:.4f}"
            )
        inference_seconds.append(time.perf_counter() - start)

    average_performance = round(float(np.mean(accuracy_matrix[-1])), 2)
    if args["num_tasks"] > 1:
        forgetting = [
            accuracy_matrix[-1, old_task] - accuracy_matrix[old_task, old_task]
            for old_task in range(args["num_tasks"] - 1)
        ]
        average_forgetting = round(float(np.mean(forgetting)), 2)
    else:
        average_forgetting = 0.0

    result = {
        "dataset": args["dataset"],
        "method": args["method"],
        "seed": args["seed"],
        "num_tasks": args["num_tasks"],
        "average_performance": average_performance,
        "average_forgetting": average_forgetting,
        "accuracy_matrix": accuracy_matrix.tolist(),
        "training_seconds_per_task": training_seconds,
        "inference_seconds_per_task": inference_seconds,
        "hyperparameters": {
            "num_epochs": args["num_epochs"],
            "lr": args["lr"],
            "weight_decay": args["weight_decay"],
            "patience": args["patience"],
            "cls_balance": args["cls_balance"],
            "shots": args["shots"],
            "k_teachers": args["k_teachers"],
            "lambda_er": args["lambda_er"],
            "lambda_kd": args["lambda_kd"],
            "lambda_node": args["lambda_node"],
            "lambda_sem": args["lambda_sem"],
            "ftfer_args": args["ftfer_args"],
            "dmsg_args": args["dmsg_args"],
        },
    }
    result_path = output_dir / (
        f"{args['dataset']}_{args['method']}_seed{args['seed']}.json"
    )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Average Performance: {average_performance}")
    print(f"Average Forgetting: {average_forgetting}")
    print(f"Result: {result_path}")
    print("=" * 60)
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="Continual heterogeneous graph-learning experiments"
    )
    parser.add_argument(
        "--dataset",
        choices=["DBLP", "IMDB", "Freebase"],
        default="DBLP",
    )
    parser.add_argument("--method", choices=METHODS, default="hero")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-tasks", dest="num_tasks", type=int, default=2)
    parser.add_argument("--num-epochs", dest="num_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--log-every", dest="log_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.001)
    parser.add_argument("--hidden-units", dest="hidden_units", type=int, default=8)
    parser.add_argument("--num-heads", dest="num_heads", type=int, nargs="+", default=[8])
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--memory-strength", dest="memory_strength", type=float, default=100)
    parser.add_argument("--lambda-l", dest="lambda_l", type=float, default=100)
    parser.add_argument("--lambda-t", dest="lambda_t", type=float, default=100)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambda-er", dest="lambda_er", type=float, default=1.0)
    parser.add_argument("--lambda-kd", dest="lambda_kd", type=float, default=1.0)
    parser.add_argument("--lambda-node", dest="lambda_node", type=float, default=1.0)
    parser.add_argument("--lambda-sem", dest="lambda_sem", type=float, default=1.0)
    parser.add_argument("--shots", type=int, default=10)
    parser.add_argument("--k-teachers", dest="k_teachers", type=int, default=3)
    parser.add_argument("--budget-ots", dest="budget_ots", type=int, default=200)
    parser.add_argument(
        "--ergnn-args",
        dest="ergnn_args",
        type=str2dict,
        default={"budget": [50], "d": [0.5], "sampler": ["CM"]},
    )
    parser.add_argument(
        "--ftfer-args",
        dest="ftfer_args",
        type=str2dict,
        default={"budget": [50], "sampler": ["mix"], "beta": [0.5]},
    )
    parser.add_argument(
        "--dmsg-args",
        dest="dmsg_args",
        type=str2dict,
        default={
            "sampler": "My",
            "budget": 50,
            "loss_weights": [1.0, 10.0, 1.0, 1.0],
        },
    )
    parser.add_argument("--cls-balance", dest="cls_balance", type=str_to_bool, default=False)
    parser.add_argument("--metrics", choices=["acc", "f1"], default="acc")
    parser.add_argument("--use-cache", dest="use_cache", type=str_to_bool, default=True)
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--data-path", dest="data_path", default="cache/ftf_er")
    parser.add_argument("--output-dir", dest="output_dir", default="outputs/main")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args().__dict__)
