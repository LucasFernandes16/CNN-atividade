import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


@dataclass
class Metrics:
    train_loss: List[float]
    test_loss: List[float]
    test_acc: List[float]
    per_class_acc: Dict[str, float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_limit_dataset(dataset, limit: int):
    if limit <= 0 or limit >= len(dataset):
        return dataset
    indices = list(range(limit))
    return Subset(dataset, indices)


class CustomCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_custom_loaders(dataset_name: str, data_root: str, batch_size: int, limit_train: int, limit_test: int):
    if dataset_name == "mnist":
        train_tf = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=train_tf)
        test_ds = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=test_tf)
        in_channels = 1
        classes = [str(i) for i in range(10)]
    else:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_ds = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
        test_ds = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
        in_channels = 3
        classes = list(train_ds.classes)

    train_ds = maybe_limit_dataset(train_ds, limit_train)
    test_ds = maybe_limit_dataset(test_ds, limit_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, in_channels, classes


def build_pretrained_loaders(dataset_name: str, data_root: str, batch_size: int, limit_train: int, limit_test: int):
    imagenet_norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if dataset_name == "mnist":
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            imagenet_norm,
        ])
        test_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            imagenet_norm,
        ])
        train_ds = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=train_tf)
        test_ds = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=test_tf)
        classes = [str(i) for i in range(10)]
    else:
        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            imagenet_norm,
        ])
        test_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            imagenet_norm,
        ])
        train_ds = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
        test_ds = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
        classes = list(train_ds.classes)

    train_ds = maybe_limit_dataset(train_ds, limit_train)
    test_ds = maybe_limit_dataset(test_ds, limit_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size
    return running_loss / max(total, 1)


def evaluate(model, loader, criterion, device, class_names: List[str]):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    class_correct = [0 for _ in class_names]
    class_total = [0 for _ in class_names]

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size
            correct += (preds == labels).sum().item()

            for label, pred in zip(labels, preds):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1

    per_class = {}
    for idx, name in enumerate(class_names):
        if class_total[idx] == 0:
            per_class[name] = 0.0
        else:
            per_class[name] = 100.0 * class_correct[idx] / class_total[idx]

    return (
        running_loss / max(total, 1),
        100.0 * correct / max(total, 1),
        per_class,
    )


def run_model(model, train_loader, test_loader, epochs, lr, weight_decay, device, class_names: List[str]):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss_hist = []
    test_loss_hist = []
    test_acc_hist = []
    per_class = {}

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, per_class = evaluate(model, test_loader, criterion, device, class_names)

        train_loss_hist.append(train_loss)
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)

    return Metrics(train_loss_hist, test_loss_hist, test_acc_hist, per_class)


def make_pretrained_resnet18(num_classes: int):
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_plots(dataset_name: str, custom_metrics: Metrics, pretrained_metrics: Metrics, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    epochs_custom = list(range(1, len(custom_metrics.test_acc) + 1))
    epochs_pre = list(range(1, len(pretrained_metrics.test_acc) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_custom, custom_metrics.test_acc, marker="o", label="Custom CNN")
    plt.plot(epochs_pre, pretrained_metrics.test_acc, marker="o", label="ResNet18 (transfer)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy - {dataset_name.upper()}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_accuracy_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_custom, custom_metrics.train_loss, marker="o", label="Custom CNN train")
    plt.plot(epochs_custom, custom_metrics.test_loss, marker="o", label="Custom CNN test")
    plt.plot(epochs_pre, pretrained_metrics.train_loss, marker="o", label="ResNet18 train")
    plt.plot(epochs_pre, pretrained_metrics.test_loss, marker="o", label="ResNet18 test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss - {dataset_name.upper()}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_loss_curve.png"), dpi=150)
    plt.close()


def run_experiment_for_dataset(
    dataset_name: str,
    data_root: str,
    output_dir: str,
    device,
    batch_size: int,
    custom_epochs: int,
    pretrained_epochs: int,
    custom_lr: float,
    pretrained_lr: float,
    weight_decay: float,
    dropout: float,
    limit_train: int,
    limit_test: int,
):
    print(f"\n=== Dataset: {dataset_name.upper()} ===")

    custom_train, custom_test, in_channels, class_names = build_custom_loaders(
        dataset_name, data_root, batch_size, limit_train, limit_test
    )
    custom_model = CustomCNN(in_channels=in_channels, num_classes=len(class_names), dropout=dropout).to(device)

    print("Training custom CNN...")
    custom_metrics = run_model(
        custom_model,
        custom_train,
        custom_test,
        custom_epochs,
        custom_lr,
        weight_decay,
        device,
        class_names,
    )

    pre_train, pre_test, class_names_pre = build_pretrained_loaders(
        dataset_name, data_root, batch_size, limit_train, limit_test
    )
    pretrained_model = make_pretrained_resnet18(num_classes=len(class_names_pre)).to(device)

    print("Fine-tuning ResNet18 (only classifier)...")
    pretrained_metrics = run_model(
        pretrained_model,
        pre_train,
        pre_test,
        pretrained_epochs,
        pretrained_lr,
        weight_decay,
        device,
        class_names_pre,
    )

    save_plots(dataset_name, custom_metrics, pretrained_metrics, output_dir)

    return {
        "custom": {
            "final_test_acc": custom_metrics.test_acc[-1],
            "final_test_loss": custom_metrics.test_loss[-1],
            "per_class_acc": custom_metrics.per_class_acc,
            "epochs": custom_epochs,
        },
        "pretrained_resnet18": {
            "final_test_acc": pretrained_metrics.test_acc[-1],
            "final_test_loss": pretrained_metrics.test_loss[-1],
            "per_class_acc": pretrained_metrics.per_class_acc,
            "epochs": pretrained_epochs,
        },
    }

def main():
    parser = argparse.ArgumentParser(description="Executa a atividade de comparacao de CNNs em MNIST e CIFAR10")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--datasets", nargs="+", default=["mnist", "cifar10"], choices=["mnist", "cifar10"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--custom-epochs", type=int, default=5)
    parser.add_argument("--pretrained-epochs", type=int, default=5)
    parser.add_argument("--custom-lr", type=float, default=1e-3)
    parser.add_argument("--pretrained-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--limit-train", type=int, default=0, help="Limita amostras de treino para execucoes rapidas")
    parser.add_argument("--limit-test", type=int, default=0, help="Limita amostras de teste para execucoes rapidas")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for dataset_name in args.datasets:
        all_results[dataset_name] = run_experiment_for_dataset(
            dataset_name=dataset_name,
            data_root=args.data_root,
            output_dir=args.output_dir,
            device=device,
            batch_size=args.batch_size,
            custom_epochs=args.custom_epochs,
            pretrained_epochs=args.pretrained_epochs,
            custom_lr=args.custom_lr,
            pretrained_lr=args.pretrained_lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            limit_train=args.limit_train,
            limit_test=args.limit_test,
        )

    json_path = os.path.join(args.output_dir, "results_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Resumo salvo em: {json_path}")
    print(f"Graficos salvos em: {args.output_dir}")


if __name__ == "__main__":
    main()
