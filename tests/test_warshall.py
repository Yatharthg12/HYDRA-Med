import numpy as np

from urop_healthgraph.warshall import EXPECTED_CLOSURE, warshall_iterations


def test_warshall_final_closure_and_iterations() -> None:
    result = warshall_iterations()
    assert result["verified"] is True
    assert len(result["iterations"]) == 6
    assert np.array_equal(
        np.asarray(result["iterations"][-1]["matrix"]), EXPECTED_CLOSURE
    )
    assert result["time_complexity"] == "O(V^3)"
    assert [step["step_index"] for step in result["iterations"]] == list(range(6))
    assert [step["intermediate_node"] for step in result["iterations"]] == [
        None,
        "Patient",
        "Disease",
        "Medication",
        "Lab Test",
        "Complication",
    ]
    assert result["iterations"][0]["matrix"] == [
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    assert {
        (cell["row"], cell["column"])
        for cell in result["iterations"][2]["new_cells"]
    } == {(0, 2), (0, 4), (3, 2), (3, 4)}


def test_warshall_dashboard_static_initialization_contract() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    template = (root / "web/templates/warshall.html").read_text(encoding="utf-8")
    script = (root / "web/static/js/dashboard.js").read_text(encoding="utf-8")
    for identifier in (
        "warshall-step-slider",
        "warshall-prev",
        "warshall-next",
        "warshall-counter",
        "warshall-matrix",
        "warshall-intermediate",
        "warshall-description",
        "warshall-calculations",
        "warshall-graph",
        "warshall-reset",
        "warshall-autoplay",
    ):
        assert f'id="{identifier}"' in template
    assert 'value="0"' in template
    assert "Step 0 of 5" in template
    assert 'id="warshall-prev"' in template and "disabled" in template
    assert 'slider.addEventListener("input"' in script
    assert 'slider.addEventListener("change"' in script
    assert "renderStep(0);" in script
