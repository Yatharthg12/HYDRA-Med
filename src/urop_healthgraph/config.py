"""Central configuration for experiments and the Flask dashboard."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class ExperimentConfig:
    """All reproducibility, path, model, and graph parameters."""

    dataset: str = "reduced"
    random_seed: int = 42
    reduced_rows: int = 20_000
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    rare_category_threshold: float = 0.005
    graph_neighbors: int = 8
    graph_similarity_threshold: float = 0.35
    gcn_hidden_size: int = 32
    gcn_dropout: float = 0.35
    gcn_learning_rate: float = 0.01
    gcn_weight_decay: float = 5e-4
    gcn_max_epochs: int = 100
    gcn_patience: int = 12
    bootstrap_replicates: int = 1_000
    bootstrap_seed: int = 42_042
    gcn_stability_seeds: tuple[int, ...] = (42, 52, 62, 72, 82)
    robustness_trial_seeds: tuple[int, ...] = (
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
    )
    pca_component_grid: tuple[int, ...] = (25, 50, 75, 100, 150)
    graph_ablation_seed: int = 4_242
    edge_removal_levels: tuple[float, ...] = (0.05, 0.10, 0.20)
    noise_edge_levels: tuple[float, ...] = (0.05, 0.10)
    project_root: Path = field(default=PROJECT_ROOT)

    def __post_init__(self) -> None:
        if self.dataset not in {"reduced", "full"}:
            raise ValueError("dataset must be 'reduced' or 'full'")
        if abs(self.train_ratio + self.validation_ratio + self.test_ratio - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")

    @property
    def raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def dataset_path(self) -> Path:
        if self.dataset == "full":
            return self.raw_dir / "diabetic_data.csv"
        return self.processed_dir / "diabetic_data_reduced.csv"

    @property
    def mapping_path(self) -> Path:
        return self.raw_dir / "IDS_mapping.csv"

    @property
    def split_path(self) -> Path:
        return self.processed_dir / "split_assignments.csv"

    def ensure_directories(self) -> None:
        """Create generated-data and artifact directories."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        for name in ("models", "metrics", "graphs", "predictions", "figures"):
            (self.artifacts_dir / name).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["project_root"] = str(self.project_root)
        return result
