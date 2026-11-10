"""Train, validate and evaluate the glass-vial CNN from notebook 01_training.

Run from the repository root:
    python -m vial_classifier.modeling.train
"""

from copy import deepcopy
import json

from loguru import logger
import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
from torch import nn, optim

from vial_classifier.config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    CLASS_NAMES,
    CLASS_NAMES_PATH,
    EARLY_STOP_PATIENCE,
    EXPERIMENT_NAME,
    IMAGE_SIZE,
    IMAGE_STATISTICS_PATH,
    LEARNING_RATE,
    MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_STATE_DIR,
    MLFLOW_TRACKING_URI,
    MODEL_RESULTS_PATH,
    NUM_CLASSES,
    NUM_EPOCHS,
    PYTORCH_MODEL_PATH,
    RANDOM_STATE,
    SCHEDULER_PATIENCE,
    TEST_RATIO,
    THRESHOLD,
    TRAIN_RATIO,
    VAL_RATIO,
)
from vial_classifier.dataset import (
    create_dataloaders,
    load_image_dataset,
    pull_data_from_dvc,
    split_dataset,
)
from vial_classifier.features import calculate_image_statistics, count_classes


class SimpleCNN(nn.Module):
    """Notebook CNN: 64x64 RGB image -> one binary-classification logit."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # Raw logit: BCEWithLogitsLoss applies sigmoid internally.


def _run_epoch(model, loader, criterion, device, optimizer=None):
    """One training or validation pass; only update weights during training."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_labels, all_preds = [], []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device).reshape(-1, 1)
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = (torch.sigmoid(logits) >= THRESHOLD).int()
            all_preds.extend(predictions.detach().cpu().flatten().tolist())
            all_labels.extend(labels.detach().cpu().int().flatten().tolist())

    if not all_labels:
        raise ValueError("A DataLoader contains no images; check dataset preparation.")

    return {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


def train_cnn_model(
    train_loader,
    val_loader,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    device=None,
    random_seed=RANDOM_STATE,
):
    """Train using weighted BCE loss, LR scheduling and validation early stopping."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    model = SimpleCNN().to(device)

    train_labels = train_loader.dataset.labels
    positives = sum(int(label) == 1 for label in train_labels)
    negatives = len(train_labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Training requires both good (0) and defective (1) images.")

    pos_weight = torch.tensor([negatives / positives], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=SCHEDULER_PATIENCE
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "train_f1": [], "val_f1": [], "val_f1_macro": [],
        "best_val_f1_macro": -1.0,
    }
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["f1_macro"])

        for split, metrics in (("train", train_metrics), ("val", val_metrics)):
            for name, value in metrics.items():
                key = f"{split}_{name}"
                if key in history:
                    history[key].append(float(value))
                if mlflow.active_run():
                    mlflow.log_metric(key, float(value), step=epoch)

        logger.info(
            "Epoch {}/{} | train loss {:.4f}, F1 {:.4f} | val loss {:.4f}, macro-F1 {:.4f}",
            epoch + 1, num_epochs, train_metrics["loss"], train_metrics["f1"],
            val_metrics["loss"], val_metrics["f1_macro"],
        )

        if val_metrics["f1_macro"] > history["best_val_f1_macro"]:
            history["best_val_f1_macro"] = float(val_metrics["f1_macro"])
            best_weights = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                logger.info("Early stopping after {} epochs without improvement", EARLY_STOP_PATIENCE)
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model, history


def _predict_probabilities(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            logits = model(images.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().flatten().tolist())
            labels.extend(torch.as_tensor(batch_labels).int().flatten().tolist())
    if not labels:
        raise ValueError("Evaluation DataLoader is empty.")
    return np.asarray(probabilities), np.asarray(labels)


def tune_threshold(model, val_loader, device=None, threshold_range=None):
    """Select the probability cutoff that maximises validation macro-F1."""
    device = torch.device(device or next(model.parameters()).device)
    probabilities, labels = _predict_probabilities(model, val_loader, device)
    if threshold_range is None:
        threshold_range = np.arange(0.025, 0.976, 0.025)
    best_threshold, best_score = THRESHOLD, -1.0
    for candidate in threshold_range:
        preds = (probabilities >= candidate).astype(int)
        score = f1_score(labels, preds, average="macro", zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(candidate), float(score)
    if mlflow.active_run():
        mlflow.log_param("optimal_threshold", best_threshold)
        mlflow.log_metric("optimal_threshold_macro_f1", best_score)
    return best_threshold


def evaluate_cnn_model(model, test_loader, device=None, threshold=THRESHOLD):
    """Evaluate once on held-out test data using the validation-selected threshold."""
    device = torch.device(device or next(model.parameters()).device)
    probabilities, labels = _predict_probabilities(model, test_loader, device)
    predictions = (probabilities >= threshold).astype(int)
    results = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities))
            if len(np.unique(labels)) == 2 else None
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            labels, predictions, labels=[0, 1], output_dict=True, zero_division=0
        ),
    }
    if mlflow.active_run():
        for name in ("accuracy", "f1_score", "precision", "recall", "roc_auc"):
            if results[name] is not None:
                mlflow.log_metric(f"test_{name}", results[name])
    return results


def setup_mlflow():
    MLFLOW_STATE_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_LOCATION)
    mlflow.set_experiment(EXPERIMENT_NAME)


def main():
    """Load images, split, train, tune, evaluate and save the model."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_mlflow()

    pull_data_from_dvc()
    image_dataset = load_image_dataset()
    if not image_dataset:
        raise ValueError("No images found. Download the dataset before running training.")
    labels = [label for _, label in image_dataset]
    class_distribution = count_classes(labels)
    logger.info("Class distribution: {}", class_distribution)
    CLASS_NAMES_PATH.write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")

    train_set, val_set, test_set = split_dataset(
        image_dataset,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_STATE,
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        train_set, val_set, test_set, batch_size=BATCH_SIZE
    )
    if not all(len(loader.dataset) for loader in (train_loader, val_loader, test_loader)):
        raise ValueError("Train, validation and test splits must all contain images.")
    image_statistics = calculate_image_statistics(train_loader)
    IMAGE_STATISTICS_PATH.write_text(json.dumps(image_statistics, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on {}", device)
    with mlflow.start_run(run_name="cnn_training"):
        mlflow.log_params({
            "model_architecture": "SimpleCNN", "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS, "learning_rate": LEARNING_RATE,
            "initial_threshold": THRESHOLD, "num_classes": NUM_CLASSES,
            "image_size": str(IMAGE_SIZE), "random_seed": RANDOM_STATE,
        })
        model, history = train_cnn_model(train_loader, val_loader, device=device)
        optimal_threshold = tune_threshold(model, val_loader, device=device)
        results = evaluate_cnn_model(model, test_loader, device=device,
                                     threshold=optimal_threshold)
        torch.save({
            "model_state_dict": model.state_dict(),
            "architecture": SimpleCNN.__name__,
            "input_shape": (3, *IMAGE_SIZE),
            "output_shape": (1,),
            "threshold": optimal_threshold,
            "num_classes": NUM_CLASSES,
        }, PYTORCH_MODEL_PATH)
        MODEL_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(PYTORCH_MODEL_PATH))
        mlflow.log_artifact(str(MODEL_RESULTS_PATH))
        logger.success("Saved model to {}", PYTORCH_MODEL_PATH)
if __name__ == "__main__":
    main()