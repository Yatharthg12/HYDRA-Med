"""Portable pure-PyTorch two-layer graph convolutional network."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from torch import nn

from .evaluation import classification_metrics, curve_points, select_threshold


class GraphConvolution(nn.Module):
    """A graph convolution using a pre-normalized sparse adjacency matrix."""

    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_features, output_features))
        self.bias = nn.Parameter(torch.zeros(output_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        support = features @ self.weight
        return torch.sparse.mm(adjacency, support) + self.bias


class TwoLayerGCN(nn.Module):
    """Two graph-convolution layers with ReLU and dropout."""

    def __init__(self, input_features: int, hidden_size: int = 32, dropout: float = 0.35):
        super().__init__()
        self.first = GraphConvolution(input_features, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.second = GraphConvolution(hidden_size, 1)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.first(features, adjacency))
        hidden = self.dropout(hidden)
        return self.second(hidden, adjacency).squeeze(1)


def scipy_to_torch_sparse(matrix: sparse.spmatrix) -> torch.Tensor:
    """Convert a SciPy sparse matrix to a coalesced float32 tensor."""
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    # Indices are constructed directly from a valid SciPy COO matrix. Opting out
    # explicitly avoids PyTorch's warning about the implicit default.
    torch.sparse.check_sparse_tensor_invariants.disable()
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=coo.shape,
        dtype=torch.float32,
        check_invariants=False,
    ).coalesce()


def _probabilities(
    model: TwoLayerGCN, features: torch.Tensor, adjacency: torch.Tensor
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(features, adjacency)).cpu().numpy()


def train_gcn(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_adjacency: sparse.spmatrix,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    validation_adjacency: sparse.spmatrix,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    test_adjacency: sparse.spmatrix,
    *,
    hidden_size: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    maximum_epochs: int,
    patience: int,
    random_seed: int,
    model_path: Path,
) -> tuple[TwoLayerGCN, dict[str, Any], np.ndarray, np.ndarray]:
    """Train with weighted BCE and early stopping on validation PR-AUC."""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    train_x = torch.from_numpy(np.asarray(train_features, dtype=np.float32))
    validation_x = torch.from_numpy(np.asarray(validation_features, dtype=np.float32))
    test_x = torch.from_numpy(np.asarray(test_features, dtype=np.float32))
    train_y = torch.from_numpy(np.asarray(train_labels, dtype=np.float32))
    train_adj = scipy_to_torch_sparse(train_adjacency)
    validation_adj = scipy_to_torch_sparse(validation_adjacency)
    test_adj = scipy_to_torch_sparse(test_adjacency)

    model = TwoLayerGCN(train_x.shape[1], hidden_size, dropout)
    positives = max(1.0, float(train_y.sum()))
    negatives = max(1.0, float(len(train_y) - train_y.sum()))
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_pr_auc = -np.inf
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_x, train_adj)
        loss = loss_function(logits, train_y)
        loss.backward()
        optimizer.step()

        validation_probability = _probabilities(model, validation_x, validation_adj)
        validation_metrics = classification_metrics(
            validation_labels, validation_probability, 0.5
        )
        validation_pr = float(validation_metrics["pr_auc"] or 0.0)
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(loss.item()),
                "validation_pr_auc": validation_pr,
                "validation_roc_auc": validation_metrics["roc_auc"],
            }
        )
        if validation_pr > best_pr_auc + 1e-5:
            best_pr_auc = validation_pr
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("GCN training did not produce a checkpoint")
    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - started
    validation_probabilities = _probabilities(model, validation_x, validation_adj)
    threshold = select_threshold(validation_labels, validation_probabilities)
    inference_started = time.perf_counter()
    test_probabilities = _probabilities(model, test_x, test_adj)
    inference_seconds = time.perf_counter() - inference_started

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_features": int(train_x.shape[1]),
            "hidden_size": hidden_size,
            "dropout": dropout,
        },
        model_path,
    )
    result = {
        "name": "Graph Convolutional Network",
        "slug": "gcn",
        "selected_hyperparameters": {
            "hidden_size": hidden_size,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "pos_weight": negatives / positives,
            "maximum_epochs": maximum_epochs,
            "early_stopping_patience": patience,
        },
        "validation_selected_threshold": threshold,
        "validation_metrics": classification_metrics(
            validation_labels, validation_probabilities, threshold
        ),
        "test_metrics": classification_metrics(
            test_labels, test_probabilities, threshold
        ),
        "test_metrics_at_0_5": classification_metrics(
            test_labels, test_probabilities, 0.5
        ),
        "curves": curve_points(test_labels, test_probabilities),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "epochs_completed": len(history),
        "best_epoch": int(np.argmax([row["validation_pr_auc"] for row in history]) + 1),
        "history": history,
    }
    return model, result, validation_probabilities, test_probabilities


def predict_gcn(
    model: TwoLayerGCN, features: np.ndarray, adjacency: sparse.spmatrix
) -> np.ndarray:
    """Evaluate a trained GCN on an alternative graph structure."""
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    adj = scipy_to_torch_sparse(adjacency)
    return _probabilities(model, x, adj)
