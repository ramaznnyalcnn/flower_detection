from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.data_utils import write_json


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, models, transforms
    except ImportError as exc:  # pragma: no cover - optional dependency branch
        raise ImportError("torch and torchvision are required for CNN training.") from exc
    return torch, nn, DataLoader, datasets, models, transforms


def get_device(prefer_cuda: bool = True):
    torch, *_ = _require_torch()
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms(image_size: int = 224, train: bool = False):
    _, _, _, _, _, transforms = _require_torch()
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.15),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_resnet50(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True):
    torch, nn, _, _, models, _ = _require_torch()
    weights = None
    if pretrained:
        try:
            weights = models.ResNet50_Weights.DEFAULT
        except AttributeError:  # pragma: no cover - old torchvision branch
            weights = "DEFAULT"

    try:
        model = models.resnet50(weights=weights)
    except Exception:
        if pretrained:
            print("Pretrained weights unavailable; falling back to random initialization.")
        model = models.resnet50(weights=None)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, num_classes),
    )
    return model


def unfreeze_last_resnet_block(model) -> None:
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def make_dataloaders(data_dir: str | Path, image_size: int, batch_size: int, num_workers: int = 2):
    _, _, DataLoader, datasets, _, _ = _require_torch()
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"CNN training expects train/ and val/ under {data_dir}")

    train_dataset = datasets.ImageFolder(train_dir, transform=build_transforms(image_size, train=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=build_transforms(image_size, train=False))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_dataset.classes, train_dataset.class_to_idx


def _run_epoch(model, loader, criterion, optimizer, device, train: bool) -> dict:
    torch, *_ = _require_torch()
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * inputs.size(0)
        correct += int((outputs.argmax(dim=1) == labels).sum().item())
        total += int(inputs.size(0))

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "samples": total,
    }


def train_cnn(
    data_dir: str | Path,
    output_path: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-4,
    patience: int = 5,
    pretrained: bool = True,
    fine_tune: bool = False,
    prefer_cuda: bool = True,
    num_workers: int = 2,
) -> dict:
    torch, nn, _, _, _, _ = _require_torch()
    device = get_device(prefer_cuda=prefer_cuda)
    train_loader, val_loader, classes, class_to_idx = make_dataloaders(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = build_resnet50(len(classes), pretrained=pretrained, freeze_backbone=not fine_tune).to(device)
    if fine_tune:
        unfreeze_last_resnet_block(model)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_acc = -1.0
    best_epoch = 0
    stale_epochs = 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}: train_acc={row['train_accuracy']:.4f} "
            f"val_acc={row['val_accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_type": "cnn",
                    "architecture": "resnet50",
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "class_to_idx": class_to_idx,
                    "image_size": image_size,
                    "pretrained": pretrained,
                    "fine_tune": fine_tune,
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_acc,
                },
                output_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    elapsed = time.perf_counter() - start
    summary = {
        "model": "resnet50",
        "checkpoint": str(output_path),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "elapsed_seconds": elapsed,
        "history": history,
        "classes": classes,
    }
    write_json(output_path.with_suffix(".history.json"), summary)
    return summary


def load_cnn_checkpoint(path: str | Path, prefer_cuda: bool = False):
    torch, *_ = _require_torch()
    device = get_device(prefer_cuda=prefer_cuda)
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_type") != "cnn":
        raise ValueError(f"Not a CNN checkpoint: {path}")

    model = build_resnet50(
        num_classes=len(checkpoint["classes"]),
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, device


def predict_cnn(image_path: str | Path, checkpoint_path: str | Path, top_k: int = 5, prefer_cuda: bool = False):
    torch, *_ = _require_torch()
    model, checkpoint, device = load_cnn_checkpoint(checkpoint_path, prefer_cuda=prefer_cuda)
    transform = build_transforms(checkpoint.get("image_size", 224), train=False)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]

    classes = np.asarray(checkpoint["classes"])
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    return [
        {"label": str(classes[index]), "score": float(probabilities[index])}
        for index in top_indices
    ]

