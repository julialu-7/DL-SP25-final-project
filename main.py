from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import (
    JEPAWorldModel,
    JEPAWorldModelV1,
    JEPAWorldModelV2,
    JEPAWorldModelV3,
)
import glob
import copy


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL25SP"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {
        "normal": probe_val_normal_ds,
        "wall": probe_val_wall_ds,
    }

    return probe_train_ds, probe_val_ds


MODEL_VARIANTS = [
    ("Base", JEPAWorldModel),
    ("MomentumJEPA_V1", JEPAWorldModelV1),
    ("VICRegJEPA_V2", JEPAWorldModelV2),
    ("ResNetJEPA_V3", JEPAWorldModelV3),
]


def evaluate_model(device, model_cls, model_name, probe_train_ds, probe_val_ds):
    print(f"\n--- Evaluating {model_name} ---")
    # Initialize model
    model = model_cls().to(device)

    # Compute and print total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name} | Trainable Params: {total_params:,}")

    # Set up probing evaluator
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    # Train a new prober and evaluate
    prober = evaluator.train_pred_prober()
    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{model_name} | {probe_attr} loss: {loss}")


if __name__ == "__main__":
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)

    # Benchmark each model variant
    for name, cls in MODEL_VARIANTS:
        evaluate_model(device, cls, name, probe_train_ds, probe_val_ds)
