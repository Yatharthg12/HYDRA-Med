import pandas as pd

from urop_healthgraph.config import ExperimentConfig
from urop_healthgraph.split_manager import (
    assert_disjoint_patients,
    create_patient_splits,
    split_summary,
)


def test_patient_groups_are_disjoint() -> None:
    frame = pd.DataFrame(
        {
            "encounter_id": [str(index) for index in range(80)],
            "patient_nbr": [str(index // 2) for index in range(80)],
            "target": [int((index // 2) % 5 == 0) for index in range(80)],
        }
    )
    assignments = create_patient_splits(frame, ExperimentConfig())
    assert_disjoint_patients(frame, assignments)
    summary = split_summary(frame, assignments)
    assert summary["disjoint"] is True
    assert not any(summary["patient_overlap_counts"].values())
    assert set(assignments) == {"train", "validation", "test"}
