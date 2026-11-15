import os
import random
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from config import Config


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DataManager:
    def __init__(self, config: Config):
        self.config = config
        self.class_names = None

    def get_transforms(self) -> Dict[str, transforms.Compose]:
        transform_dict = {
            "train": transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Lambda(
                        lambda img: img.convert("RGB") if img.mode != "RGB" else img
                    ),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        }
        return transform_dict

    def _loader_kwargs(self, shuffle: bool) -> Dict:
        return {
            "batch_size": self.config.batch_size,
            "shuffle": shuffle,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
        }

    def load_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        transforms_dict = self.get_transforms()

        # Load datasets
        train_data = datasets.ImageFolder(
            os.path.join(self.config.data_dir, "Train"),
            transform=transforms_dict["train"],
        )
        val_data = datasets.ImageFolder(
            os.path.join(self.config.data_dir, "Validation"),
            transform=transforms_dict["val"],
        )
        test_data = datasets.ImageFolder(
            os.path.join(self.config.data_dir, "Test"), transform=transforms_dict["val"]
        )

        self.class_names = train_data.classes

        # Create data loaders
        train_loader = DataLoader(train_data, **self._loader_kwargs(shuffle=True))
        val_loader = DataLoader(val_data, **self._loader_kwargs(shuffle=False))
        test_loader = DataLoader(test_data, **self._loader_kwargs(shuffle=False))

        return train_loader, val_loader, test_loader

    def load_secondary_test_loader(
        self,
    ) -> Optional[Tuple[DataLoader, List[str]]]:
        """Load the optional second Kaggle dataset used for extra testing."""
        secondary_dir = self.config.test_data_dir
        if not secondary_dir or not os.path.isdir(secondary_dir):
            print(f"Secondary test dataset not found at {secondary_dir}; skipping.")
            return None

        try:
            secondary_data = datasets.ImageFolder(
                secondary_dir, transform=self.get_transforms()["val"]
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Could not load secondary test dataset: {exc}")
            return None

        print(f"Secondary test dataset samples: {len(secondary_data)}")
        loader = DataLoader(secondary_data, **self._loader_kwargs(shuffle=False))
        return loader, secondary_data.classes


class FocalLoss(nn.Module):
    """Focal loss for hard-example mining.

    With the balanced 12K dataset, ``alpha`` is uniform (a no-op); the useful part
    is the ``(1 - pt) ** gamma`` factor, which down-weights easy, well-classified
    samples so training keeps focusing on the confused ones.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(inputs, dim=1)
        ce_loss = F.nll_loss(logp, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()


class EarlyStopping:
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        save_path: str = "best_model.pt",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """Record the score and save the model if it improved. Returns True if saved."""
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
            return True
        if val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        self.best_score = val_score
        self.save_checkpoint(model)
        self.counter = 0
        return True

    def save_checkpoint(self, model: nn.Module) -> None:
        """Save model checkpoint"""
        torch.save(model.state_dict(), self.save_path)


class MaskDetector:
    def __init__(self, config: Config, num_classes: int):
        self.config = config
        self.num_classes = num_classes
        self.device = torch.device(config.device)
        self.model = self._build_model()

    def _build_model(self) -> nn.Module:
        # Load pretrained MobileNetV3
        model = models.mobilenet_v3_large(weights="MobileNet_V3_Large_Weights.DEFAULT")

        # Freeze feature extractor
        for param in model.features.parameters():
            param.requires_grad = False

        # Create custom classifier
        model.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(),
            nn.Dropout(p=self.config.dropout_rate),
            nn.Linear(1280, self.config.intermediate_features),
            nn.Hardswish(),
            nn.Dropout(p=self.config.dropout_rate),
            nn.Linear(self.config.intermediate_features, self.num_classes),
        )

        return model.to(self.device)

    def print_model_summary(self) -> None:
        """Print a compact layer/parameter summary without extra dependencies."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(self.model.classifier)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")


class Trainer:
    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        # Loss and optimizer
        self.criterion = FocalLoss(
            alpha=config.focal_alpha, gamma=config.focal_gamma
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            save_path=config.model_save_path,
        )

        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        for epoch in range(self.config.epochs):
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation phase
            val_loss, val_acc = self.validate_epoch(val_loader)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Learning rate scheduling
            self.scheduler.step(val_acc)

            print(f"Epoch [{epoch + 1}/{self.config.epochs}]")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Early stopping check
            saved = self.early_stopping(val_acc, self.model)
            if saved:
                print(
                    f"New best model saved! Validation Accuracy: {val_acc:.4f} "
                    f"(lr={self.optimizer.param_groups[0]['lr']:g})"
                )
            else:
                print("No improvement")

            if self.early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        # Load best model
        self.model.load_state_dict(
            torch.load(
                self.config.model_save_path,
                map_location=self.device,
                weights_only=True,
            )
        )
        return self.history

    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """Plot loss and accuracy curves for the run."""
        epochs = range(1, len(self.history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(epochs, self.history["train_loss"], label="Train Loss", marker="o")
        ax1.plot(epochs, self.history["val_loss"], label="Validation Loss", marker="s")
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, self.history["train_acc"], label="Train Accuracy", marker="o")
        ax2.plot(
            epochs, self.history["val_acc"], label="Validation Accuracy", marker="s"
        )
        ax2.set_title("Training and Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
            print(f"Training curves saved to {save_path}")
        plt.close(fig)


class Evaluator:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def evaluate(self, test_loader: DataLoader, class_names: List[str]) -> Dict:
        self.model.eval()
        all_preds = []
        all_labels = []
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if batch_idx % 20 == 0:
                    print(
                        f"Processed {batch_idx * len(images)}/{len(test_loader.dataset)} images"
                    )

        accuracy = 100 * correct / total

        # Generate reports
        results = {
            "accuracy": accuracy,
            "predictions": all_preds,
            "true_labels": all_labels,
        }

        return results

    def plot_confusion_matrix(
        self,
        results: Dict,
        class_names: List[str],
        title: str,
        save_path: Optional[str] = None,
    ) -> None:
        cm = confusion_matrix(results["true_labels"], results["predictions"])
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(image, ax=ax, label="Count")

        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted Label",
            ylabel="True Label",
            title=f"{title}\nAccuracy: {results['accuracy']:.2f}%",
        )

        threshold = cm.max() / 2 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        plt.close(fig)

    def print_detailed_results(self, results: Dict, class_names: List[str]) -> None:
        print(f"\nAccuracy: {results['accuracy']:.2f}%")
        print("\nClassification Report:")
        print(
            classification_report(
                results["true_labels"],
                results["predictions"],
                target_names=class_names,
            )
        )


def main():
    config = Config()
    seed_everything(config.seed)

    # Initialize data manager
    data_manager = DataManager(config)
    train_loader, val_loader, test_loader = data_manager.load_datasets()

    # Initialize model
    mask_detector = MaskDetector(config, len(data_manager.class_names))
    mask_detector.print_model_summary()

    # Training
    trainer = Trainer(mask_detector.model, config)
    history = trainer.train(train_loader, val_loader)
    trainer.plot_training_curves("training_curves.png")

    # Evaluation
    evaluator = Evaluator(mask_detector.model, mask_detector.device)

    # Test on validation set
    val_results = evaluator.evaluate(val_loader, data_manager.class_names)
    evaluator.plot_confusion_matrix(
        val_results, data_manager.class_names, "Validation Results", "cm_validation.png"
    )
    evaluator.print_detailed_results(val_results, data_manager.class_names)

    # Test on test set
    test_results = evaluator.evaluate(test_loader, data_manager.class_names)
    evaluator.plot_confusion_matrix(
        test_results, data_manager.class_names, "Test Results", "cm_test.png"
    )
    evaluator.print_detailed_results(test_results, data_manager.class_names)

    # Optional secondary dataset, if it is available locally
    secondary = data_manager.load_secondary_test_loader()
    if secondary is not None:
        secondary_loader, secondary_classes = secondary
        secondary_results = evaluator.evaluate(secondary_loader, secondary_classes)
        evaluator.plot_confusion_matrix(
            secondary_results,
            secondary_classes,
            "Secondary Test Dataset Results",
            "cm_secondary.png",
        )
        evaluator.print_detailed_results(secondary_results, secondary_classes)

    # The best checkpoint was already written by EarlyStopping.
    print(f"Best model checkpoint: {config.model_save_path}")

    return mask_detector, history, test_results


if __name__ == "__main__":
    model, training_history, results = main()
