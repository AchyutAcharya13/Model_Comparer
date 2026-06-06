import json
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from model import CIFAR10CNN
from optimizers import get_optimizer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_NUM_EPOCHS = 5
DEFAULT_BATCH_SIZE = 128
DEFAULT_SUBSET_SIZE = 10000
DEFAULT_TEST_SUBSET_SIZE = 2000
DEFAULT_TRIALS = 1
DEFAULT_SEED = 42
CHECKPOINT_DIR = Path("./checkpoints")
ARTIFACT_DIR = Path("./artifacts")

FULL_TRAINSET_SIZE = 50000
FULL_TESTSET_SIZE = 10000

PAPER_PROTOCOL = {
    "epochs": 200,
    "batch_size": 128,
    "full_dataset": True,
    "use_augmentation": True,
}

PAPER_OPTIMIZER_DEFAULTS = {
    "adam": {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 0.0},
    "sgd": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4},
    "adabs": {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "final_lr": 0.1,
        "gamma": 1e-3,
        "weight_decay": 0.0,
    },
}

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _seed_worker(_worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _get_transforms(use_augmentation: bool):
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return train_transform, transform_test


def get_dataloaders(
    subset_size: int,
    test_subset_size: int,
    batch_size: int,
    seed: int,
    use_augmentation: bool,
    full_dataset: bool,
):
    transform, transform_test = _get_transforms(use_augmentation=use_augmentation)

    train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_data  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    if full_dataset:
        train_dataset = train_data
        test_dataset = test_data
    else:
        train_limit = min(len(train_data), max(1, subset_size))
        test_limit = min(len(test_data), max(1, test_subset_size))
        train_dataset = Subset(train_data, range(train_limit))
        test_dataset = Subset(test_data, range(test_limit))

    generator = torch.Generator()
    generator.manual_seed(seed)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct    += outputs.argmax(1).eq(targets).sum().item()
        total      += inputs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        correct    += outputs.argmax(1).eq(targets).sum().item()
        total      += inputs.size(0)
    return total_loss / total, correct / total


def save_checkpoint(
    model,
    optimizer_name: str,
    history: list[dict],
    epochs: int,
    subset_size: int,
    test_subset_size: int,
    trial_index: int,
    run_id: str,
) -> dict:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    checkpoint_path = CHECKPOINT_DIR / f"cifar10_{optimizer_name}_{run_id}_trial{trial_index}_{timestamp}.pt"
    latest_optimizer_path = CHECKPOINT_DIR / f"latest_{optimizer_name}.pt"
    latest_global_path = CHECKPOINT_DIR / "latest.pt"

    payload = {
        "optimizer": optimizer_name,
        "timestamp": timestamp,
        "classes": CIFAR10_CLASSES,
        "epochs": epochs,
        "subset_size": subset_size,
        "test_subset_size": test_subset_size,
        "model_state_dict": model.state_dict(),
        "history": history,
    }

    torch.save(payload, checkpoint_path)
    torch.save(payload, latest_optimizer_path)
    torch.save(payload, latest_global_path)

    return {
        "checkpoint_path": str(checkpoint_path).replace("\\", "/"),
        "latest_checkpoint_path": str(latest_optimizer_path).replace("\\", "/"),
    }


def _default_optimizer_params(optimizer_name: str) -> dict[str, Any]:
    canonical_name = optimizer_name.lower()
    if canonical_name == "adabound":
        canonical_name = "adabs"
    defaults = PAPER_OPTIMIZER_DEFAULTS.get(canonical_name, {})
    return dict(defaults)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.fmean(values)), float(statistics.pstdev(values))


def _write_json_artifact(optimizer_name: str, run_id: str, payload: dict[str, Any], trial_index: int | None = None) -> str:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if trial_index is None:
        filename = f"experiment_{optimizer_name}_{run_id}_{timestamp}.json"
    else:
        filename = f"trial_{optimizer_name}_{run_id}_t{trial_index}_{timestamp}.json"
    path = ARTIFACT_DIR / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return str(path).replace("\\", "/")


def run_training(
    optimizer_name: str,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    test_subset_size: int = DEFAULT_TEST_SUBSET_SIZE,
    save_model: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    deterministic: bool = True,
    trials: int = DEFAULT_TRIALS,
    full_dataset: bool = False,
    use_augmentation: bool = True,
    paper_protocol: bool = False,
    optimizer_params: dict[str, Any] | None = None,
) -> dict:
    canonical_optimizer = optimizer_name.lower()
    if canonical_optimizer == "adabound":
        canonical_optimizer = "adabs"

    protocol = {
        "epochs": num_epochs,
        "batch_size": batch_size,
        "full_dataset": full_dataset,
        "use_augmentation": use_augmentation,
        "subset_size": subset_size,
        "test_subset_size": test_subset_size,
    }
    if paper_protocol:
        protocol.update(PAPER_PROTOCOL)
        num_epochs = PAPER_PROTOCOL["epochs"]
        batch_size = PAPER_PROTOCOL["batch_size"]
        full_dataset = PAPER_PROTOCOL["full_dataset"]
        use_augmentation = PAPER_PROTOCOL["use_augmentation"]
        subset_size = FULL_TRAINSET_SIZE
        test_subset_size = FULL_TESTSET_SIZE

    effective_optimizer_params = _default_optimizer_params(canonical_optimizer)
    if optimizer_params:
        effective_optimizer_params.update(optimizer_params)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    trial_results = []
    trial_artifact_paths = []
    first_checkpoint_path = ""
    first_latest_checkpoint_path = ""
    representative_history = []

    for trial_index in range(1, trials + 1):
        trial_seed = seed + trial_index - 1
        seed_everything(trial_seed, deterministic=deterministic)

        train_loader, test_loader = get_dataloaders(
            subset_size=subset_size,
            test_subset_size=test_subset_size,
            batch_size=batch_size,
            seed=trial_seed,
            use_augmentation=use_augmentation,
            full_dataset=full_dataset,
        )
        model = CIFAR10CNN().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(canonical_optimizer, model.parameters(), **effective_optimizer_params)

        history = []
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = evaluate(model, test_loader, criterion)
            epoch_row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc * 100, 2),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc * 100, 2),
            }
            history.append(epoch_row)
            print(
                f"Trial {trial_index}/{trials} | Epoch {epoch}/{num_epochs}: "
                f"train_loss={train_loss:.4f}, val_acc={val_acc * 100:.2f}%"
            )

        final = history[-1]
        checkpoint_info = {
            "checkpoint_path": "",
            "latest_checkpoint_path": "",
        }
        if save_model:
            checkpoint_info = save_checkpoint(
                model,
                optimizer_name=canonical_optimizer,
                history=history,
                epochs=num_epochs,
                subset_size=subset_size,
                test_subset_size=test_subset_size,
                trial_index=trial_index,
                run_id=run_id,
            )

        trial_payload = {
            "run_id": run_id,
            "trial": trial_index,
            "optimizer": canonical_optimizer,
            "seed": trial_seed,
            "deterministic": deterministic,
            "protocol": {
                **protocol,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "full_dataset": full_dataset,
                "use_augmentation": use_augmentation,
                "subset_size": subset_size,
                "test_subset_size": test_subset_size,
                "paper_protocol": paper_protocol,
            },
            "optimizer_params": effective_optimizer_params,
            "history": history,
            "final_train_loss": final["train_loss"],
            "final_train_acc": final["train_acc"],
            "final_val_loss": final["val_loss"],
            "final_val_acc": final["val_acc"],
            "checkpoint_path": checkpoint_info["checkpoint_path"],
            "latest_checkpoint_path": checkpoint_info["latest_checkpoint_path"],
            "timestamp": datetime.now().isoformat(),
        }
        trial_artifact_path = _write_json_artifact(canonical_optimizer, run_id, trial_payload, trial_index=trial_index)
        trial_payload["artifact_path"] = trial_artifact_path
        trial_artifact_paths.append(trial_artifact_path)
        trial_results.append(trial_payload)

        if trial_index == 1:
            representative_history = history
            first_checkpoint_path = checkpoint_info["checkpoint_path"]
            first_latest_checkpoint_path = checkpoint_info["latest_checkpoint_path"]

    train_loss_values = [row["final_train_loss"] for row in trial_results]
    train_acc_values = [row["final_train_acc"] for row in trial_results]
    val_loss_values = [row["final_val_loss"] for row in trial_results]
    val_acc_values = [row["final_val_acc"] for row in trial_results]

    final_train_loss_mean, final_train_loss_std = _mean_std(train_loss_values)
    final_train_acc_mean, final_train_acc_std = _mean_std(train_acc_values)
    final_val_loss_mean, final_val_loss_std = _mean_std(val_loss_values)
    final_val_acc_mean, final_val_acc_std = _mean_std(val_acc_values)

    summary = {
        "num_trials": trials,
        "final_train_loss_mean": round(final_train_loss_mean, 4),
        "final_train_loss_std": round(final_train_loss_std, 4),
        "final_train_acc_mean": round(final_train_acc_mean, 4),
        "final_train_acc_std": round(final_train_acc_std, 4),
        "final_val_loss_mean": round(final_val_loss_mean, 4),
        "final_val_loss_std": round(final_val_loss_std, 4),
        "final_val_acc_mean": round(final_val_acc_mean, 4),
        "final_val_acc_std": round(final_val_acc_std, 4),
    }

    aggregate_payload = {
        "run_id": run_id,
        "optimizer": canonical_optimizer,
        "seed": seed,
        "deterministic": deterministic,
        "trials": trials,
        "paper_protocol": paper_protocol,
        "protocol": {
            **protocol,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "full_dataset": full_dataset,
            "use_augmentation": use_augmentation,
            "subset_size": subset_size,
            "test_subset_size": test_subset_size,
        },
        "optimizer_params": effective_optimizer_params,
        "summary": summary,
        "trial_artifact_paths": trial_artifact_paths,
        "timestamp": datetime.now().isoformat(),
    }
    aggregate_artifact_path = _write_json_artifact(canonical_optimizer, run_id, aggregate_payload)

    return {
        "optimizer": canonical_optimizer,
        "epochs": num_epochs,
        "subset_size": subset_size,
        "test_subset_size": test_subset_size,
        "batch_size": batch_size,
        "full_dataset": full_dataset,
        "use_augmentation": use_augmentation,
        "paper_protocol": paper_protocol,
        "trials": trials,
        "seed": seed,
        "deterministic": deterministic,
        "optimizer_params": effective_optimizer_params,
        "final_train_loss": summary["final_train_loss_mean"],
        "final_train_acc": summary["final_train_acc_mean"],
        "final_val_loss": summary["final_val_loss_mean"],
        "final_val_acc": summary["final_val_acc_mean"],
        "history": representative_history,
        "summary": summary,
        "trial_results": trial_results,
        "artifact_path": aggregate_artifact_path,
        "trial_artifact_paths": trial_artifact_paths,
        "checkpoint_path": first_checkpoint_path,
        "latest_checkpoint_path": first_latest_checkpoint_path,
    }
