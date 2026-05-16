from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src.data_utils import write_json

BACKBONES = {"resnet50", "efficientnet_b0", "convnext_tiny"}


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, models, transforms
    except ImportError as exc:
        raise ImportError("torch and torchvision are required for CNN training.") from exc
    return torch, nn, DataLoader, datasets, models, transforms


def get_device(prefer_cuda: bool = True):
    torch, *_ = _require_torch()
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(image_size: int = 224, train: bool = False, use_randaugment: bool = False):
    _, _, _, _, _, transforms = _require_torch()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        ops = [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if use_randaugment:
            ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
        else:
            ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03))
        ops += [
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.15),
        ]
        return transforms.Compose(ops)

    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
):
    if backbone not in BACKBONES:
        raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")

    torch, nn, _, _, models, _ = _require_torch()

    if backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        try:
            model = models.resnet50(weights=weights)
        except Exception:
            model = models.resnet50(weights=None)
        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Sequential(nn.Dropout(p=0.25), nn.Linear(model.fc.in_features, num_classes))

    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        try:
            model = models.efficientnet_b0(weights=weights)
        except Exception:
            model = models.efficientnet_b0(weights=None)
        if freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:  # convnext_tiny
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        try:
            model = models.convnext_tiny(weights=weights)
        except Exception:
            model = models.convnext_tiny(weights=None)
        if freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    return model


def build_resnet50(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True):
    """Backward-compatible wrapper."""
    return build_model("resnet50", num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)


def unfreeze_last_blocks(model, backbone: str) -> None:
    """Unfreeze the last two feature blocks and the classification head."""
    _, nn, _, _, _, _ = _require_torch()

    if backbone == "resnet50":
        for p in model.layer3.parameters():
            p.requires_grad = True
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True

    elif backbone in ("efficientnet_b0", "convnext_tiny"):
        feature_blocks = list(model.features.children())
        for block in feature_blocks[-2:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True


def unfreeze_last_resnet_block(model) -> None:
    """Backward-compatible wrapper."""
    unfreeze_last_blocks(model, "resnet50")


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def make_dataloaders(
    data_dir: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int = 2,
    use_randaugment: bool = False,
    weighted_sampler: bool = True,
):
    torch, _, DataLoader, datasets, _, _ = _require_torch()
    from torch.utils.data import WeightedRandomSampler

    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"CNN training expects train/ and val/ under {data_dir}")

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=build_transforms(image_size, train=True, use_randaugment=use_randaugment),
    )
    val_dataset = datasets.ImageFolder(val_dir, transform=build_transforms(image_size, train=False))

    if weighted_sampler:
        class_counts = Counter(train_dataset.targets)
        sample_weights = [1.0 / class_counts[t] for t in train_dataset.targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_dataset.classes, train_dataset.class_to_idx


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

def _mixup_data(x, y, alpha: float, device):
    torch, *_ = _require_torch()
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def _mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, criterion, optimizer, device, train: bool, use_mixup: bool = False) -> dict:
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

        if train and use_mixup:
            inputs, y_a, y_b, lam = _mixup_data(inputs, labels, alpha=0.4, device=device)
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = _mixup_criterion(criterion, outputs, y_a, y_b, lam)
                loss.backward()
                optimizer.step()
            correct += int((outputs.argmax(dim=1) == labels).sum().item())
        else:
            with torch.set_grad_enabled(train):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if train:
                    loss.backward()
                    optimizer.step()
            correct += int((outputs.argmax(dim=1) == labels).sum().item())

        total_loss += float(loss.item()) * inputs.size(0)
        total += int(inputs.size(0))

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "samples": total,
    }


def train_cnn(
    data_dir: str | Path,
    output_path: str | Path,
    backbone: str = "resnet50",
    image_size: int = 224,
    batch_size: int = 32,
    epochs: int = 15,
    stage1_epochs: int = 5,
    lr: float = 1e-4,
    patience: int = 5,
    pretrained: bool = True,
    fine_tune: bool = False,
    prefer_cuda: bool = True,
    num_workers: int = 2,
    use_mixup: bool = False,
    use_randaugment: bool = True,
    weighted_sampler: bool = True,
) -> dict:
    if backbone not in BACKBONES:
        raise ValueError(f"backbone must be one of {BACKBONES}")

    torch, nn, _, _, _, _ = _require_torch()
    device = get_device(prefer_cuda=prefer_cuda)

    train_loader, val_loader, classes, class_to_idx = make_dataloaders(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        use_randaugment=use_randaugment,
        weighted_sampler=weighted_sampler,
    )

    # Stage 1: frozen backbone — only head trains
    freeze = not fine_tune
    model = build_model(backbone, len(classes), pretrained=pretrained, freeze_backbone=freeze).to(device)

    # Class weights — dengesiz sınıflar için
    from collections import Counter
    cls_counts = Counter()
    for _, label_idx in train_loader.dataset.samples:
        cls_counts[label_idx] += 1
    total = sum(cls_counts.values())
    n_classes = len(classes)
    weights = torch.tensor(
        [total / (n_classes * cls_counts.get(i, 1)) for i in range(n_classes)],
        dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    history = []
    best_val_acc = -1.0
    best_epoch = 0
    stale_epochs = 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    stage2_epochs = max(epochs - stage1_epochs, 0)
    total_stages = [(stage1_epochs, lr, False)] if not fine_tune else []
    if not fine_tune and stage2_epochs > 0:
        total_stages.append((stage2_epochs, lr * 0.1, True))
    if fine_tune:
        total_stages = [(epochs, lr, True)]

    epoch_counter = 0
    for stage_epochs, stage_lr, do_unfreeze in total_stages:
        if do_unfreeze:
            unfreeze_last_blocks(model, backbone)
            print(f"  [Stage 2] Unfreezing last blocks (lr={stage_lr:.2e})")

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=stage_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(stage_epochs, 1))

        for _ in range(stage_epochs):
            epoch_counter += 1
            train_metrics = _run_epoch(model, train_loader, criterion, optimizer, device, train=True, use_mixup=use_mixup)
            val_metrics = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            scheduler.step()

            row = {
                "epoch": epoch_counter,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            history.append(row)
            print(
                f"Epoch {epoch_counter:03d}: train_acc={row['train_accuracy']:.4f} "
                f"val_acc={row['val_accuracy']:.4f}"
            )

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_epoch = epoch_counter
                stale_epochs = 0
                torch.save(
                    {
                        "model_type": "cnn",
                        "architecture": backbone,
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
                    print(f"Early stopping at epoch {epoch_counter}")
                    break

    elapsed = time.perf_counter() - start
    summary = {
        "model": backbone,
        "checkpoint": str(output_path),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "elapsed_seconds": elapsed,
        "history": history,
        "classes": classes,
    }
    write_json(output_path.with_suffix(".history.json"), summary)
    return summary


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_cnn_checkpoint(path: str | Path, prefer_cuda: bool = False):
    torch, *_ = _require_torch()
    device = get_device(prefer_cuda=prefer_cuda)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "cnn":
        raise ValueError(f"Not a CNN checkpoint: {path}")

    arch = checkpoint.get("architecture", "resnet50")
    model = build_model(arch, num_classes=len(checkpoint["classes"]), pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, device


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_cnn(
    image_path: str | Path,
    checkpoint_path: str | Path,
    top_k: int = 5,
    prefer_cuda: bool = True,
    use_tta: bool = False,
):
    if use_tta:
        return predict_cnn_tta(image_path, checkpoint_path, top_k=top_k, prefer_cuda=prefer_cuda)

    torch, *_ = _require_torch()
    model, checkpoint, device = load_cnn_checkpoint(checkpoint_path, prefer_cuda=prefer_cuda)
    transform = build_transforms(checkpoint.get("image_size", 224), train=False)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]

    classes = np.asarray(checkpoint["classes"])
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [{"label": str(classes[i]), "score": float(probs[i])} for i in top_indices]


def predict_cnn_tta(
    image_path: str | Path,
    checkpoint_path: str | Path,
    top_k: int = 5,
    prefer_cuda: bool = True,
):
    torch, _, _, _, _, transforms = _require_torch()
    model, checkpoint, device = load_cnn_checkpoint(checkpoint_path, prefer_cuda=prefer_cuda)
    image_size = checkpoint.get("image_size", 224)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    image = Image.open(image_path).convert("RGB")
    resized = transforms.Resize(int(image_size * 1.14))(image)

    # 5 crops × 2 flips = 10 augmented views
    crops = transforms.FiveCrop(image_size)(resized)
    views = []
    for crop in crops:
        views.append(to_tensor(crop))
        views.append(to_tensor(transforms.functional.hflip(crop)))

    batch = torch.stack([normalize(v) for v in views]).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1).mean(dim=0).cpu().numpy()

    classes = np.asarray(checkpoint["classes"])
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [{"label": str(classes[i]), "score": float(probs[i])} for i in top_indices]
