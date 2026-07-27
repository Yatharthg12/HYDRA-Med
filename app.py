"""Flask entry point for the HealthGraph Readmission Lab dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flask import Flask, jsonify, render_template, request, send_from_directory

from urop_healthgraph.artifacts import read_json


ARTIFACT_FILES = {
    "dataset": Path("metrics/dataset_summary.json"),
    "models": Path("metrics/model_comparison.json"),
    "bootstrap": Path("metrics/bootstrap_confidence_intervals.json"),
    "paired": Path("metrics/paired_model_differences.json"),
    "seed_stability": Path("metrics/gcn_seed_stability.json"),
    "pca_analysis": Path("metrics/pca_component_analysis.json"),
    "ablation": Path("metrics/graph_ablation_results.json"),
    "warshall": Path("graphs/warshall_iterations.json"),
    "graph": Path("graphs/sample_graph.json"),
    "graph_stats": Path("graphs/graph_statistics.json"),
    "robustness_trials": Path("metrics/robustness_trials.json"),
    "robustness_summary": Path("metrics/robustness_summary.json"),
    "cases": Path("predictions/case_records.json"),
    "manifest": Path("metrics/run_manifest.json"),
}

DOWNLOADABLE_RESULTS = {
    f"{stem}.{extension}"
    for stem in (
        "bootstrap_confidence_intervals",
        "paired_model_differences",
        "gcn_seed_stability",
        "robustness_trials",
        "robustness_summary",
        "pca_component_analysis",
        "graph_ablation_results",
    )
    for extension in ("csv", "json")
}


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    """Application factory used by production startup and route tests."""
    app = Flask(
        __name__,
        template_folder=str(ROOT / "web" / "templates"),
        static_folder=str(ROOT / "web" / "static"),
        static_url_path="/static",
    )
    app.config.from_mapping(
        ARTIFACTS_DIR=ROOT / "artifacts",
        SETUP_COMMAND="python run_experiments.py --dataset reduced",
        JSON_SORT_KEYS=False,
    )
    if test_config:
        app.config.update(test_config)

    def artifact_path(name: str) -> Path:
        return Path(app.config["ARTIFACTS_DIR"]) / ARTIFACT_FILES[name]

    def artifact(name: str, default: Any = None) -> Any:
        return read_json(artifact_path(name), default)

    def missing_artifacts(names: tuple[str, ...]) -> list[str]:
        return [name for name in names if not artifact_path(name).exists()]

    def page(
        template: str,
        required: tuple[str, ...],
        *,
        title: str,
        page_id: str,
        **context: Any,
    ):
        missing = missing_artifacts(required)
        if missing:
            return render_template(
                "setup.html",
                title="Setup required",
                page_id="setup",
                missing=missing,
                command=app.config["SETUP_COMMAND"],
            )
        return render_template(
            template, title=title, page_id=page_id, **context
        )

    @app.get("/")
    def overview():
        return page(
            "overview.html",
            ("dataset", "models", "graph_stats"),
            title="Overview",
            page_id="overview",
            dataset=artifact("dataset"),
            comparison=artifact("models"),
            graph_stats=artifact("graph_stats"),
        )

    @app.get("/dataset")
    def dataset():
        return page(
            "dataset.html",
            ("dataset",),
            title="Dataset Explorer",
            page_id="dataset",
            dataset=artifact("dataset"),
        )

    @app.get("/warshall")
    def warshall():
        return page(
            "warshall.html",
            ("warshall",),
            title="Warshall Lab",
            page_id="warshall",
            warshall=artifact("warshall"),
        )

    @app.get("/models")
    def models():
        return page(
            "models.html",
            (
                "models",
                "bootstrap",
                "paired",
                "seed_stability",
                "pca_analysis",
                "ablation",
            ),
            title="Model Comparison",
            page_id="models",
            comparison=artifact("models"),
            bootstrap=artifact("bootstrap"),
            paired=artifact("paired"),
            seed_stability=artifact("seed_stability"),
            pca_analysis=artifact("pca_analysis"),
            ablation=artifact("ablation"),
        )

    @app.get("/graph")
    def graph():
        return page(
            "graph.html",
            ("graph", "graph_stats"),
            title="Graph Explorer",
            page_id="graph",
            graph=artifact("graph"),
            graph_stats=artifact("graph_stats"),
            ablation=artifact("ablation", {}),
        )

    @app.get("/robustness")
    def robustness():
        return page(
            "robustness.html",
            ("robustness_summary", "robustness_trials"),
            title="Robustness Simulation",
            page_id="robustness",
            robustness_summary=artifact("robustness_summary"),
            robustness_trials=artifact("robustness_trials"),
        )

    @app.get("/cases")
    def cases():
        records = artifact("cases", {})
        return page(
            "cases.html",
            ("cases",),
            title="Case Explorer",
            page_id="cases",
            encounter_ids=list(records)[:250],
        )

    @app.get("/limitations")
    def limitations():
        return render_template(
            "limitations.html",
            title="Methodology & Limitations",
            page_id="limitations",
            dataset=artifact("dataset", {}),
            bootstrap=artifact("bootstrap", {}),
            seed_stability=artifact("seed_stability", {}),
            robustness_summary=artifact("robustness_summary", []),
            pca_analysis=artifact("pca_analysis", {}),
            ablation=artifact("ablation", {}),
        )

    @app.get("/docs/<path:filename>")
    def documentation(filename: str):
        return send_from_directory(ROOT / "docs", filename)

    def api_payload(name: str):
        path = artifact_path(name)
        if not path.exists():
            return (
                jsonify(
                    {
                        "status": "artifacts_missing",
                        "artifact": name,
                        "setup_command": app.config["SETUP_COMMAND"],
                    }
                ),
                503,
            )
        return jsonify(artifact(name))

    @app.get("/api/dataset")
    def api_dataset():
        return api_payload("dataset")

    @app.get("/api/models")
    def api_models():
        return api_payload("models")

    @app.get("/api/statistics")
    def api_statistics():
        missing = missing_artifacts(("bootstrap", "paired", "seed_stability"))
        if missing:
            return (
                jsonify(
                    {
                        "status": "artifacts_missing",
                        "missing_artifacts": missing,
                        "setup_command": app.config["SETUP_COMMAND"],
                    }
                ),
                503,
            )
        return jsonify(
            {
                "bootstrap_confidence_intervals": artifact("bootstrap"),
                "paired_model_differences": artifact("paired"),
                "gcn_seed_stability": artifact("seed_stability"),
            }
        )

    @app.get("/api/warshall")
    def api_warshall():
        return api_payload("warshall")

    @app.get("/api/graph")
    def api_graph():
        return api_payload("graph")

    @app.get("/api/robustness")
    def api_robustness():
        missing = missing_artifacts(("robustness_summary", "robustness_trials"))
        if missing:
            return (
                jsonify(
                    {
                        "status": "artifacts_missing",
                        "missing_artifacts": missing,
                        "setup_command": app.config["SETUP_COMMAND"],
                    }
                ),
                503,
            )
        return jsonify(
            {
                "summary": artifact("robustness_summary"),
                "trials": artifact("robustness_trials"),
            }
        )

    @app.get("/api/pca-analysis")
    def api_pca_analysis():
        return api_payload("pca_analysis")

    @app.get("/api/ablation")
    def api_ablation():
        return api_payload("ablation")

    @app.get("/downloads/results/<filename>")
    def download_result(filename: str):
        if filename not in DOWNLOADABLE_RESULTS:
            return jsonify({"status": "not_found", "filename": filename}), 404
        path = Path(app.config["ARTIFACTS_DIR"]) / "metrics" / filename
        if not path.is_file():
            return (
                jsonify(
                    {
                        "status": "artifacts_missing",
                        "filename": filename,
                        "setup_command": app.config["SETUP_COMMAND"],
                    }
                ),
                503,
            )
        return send_from_directory(path.parent, path.name, as_attachment=True)

    @app.get("/api/cases")
    def api_case_search():
        records = artifact("cases")
        if records is None:
            return api_payload("cases")
        query = request.args.get("q", "").strip().lower()
        identifiers = [
            encounter
            for encounter in records
            if not query or query in encounter.lower()
        ][:50]
        return jsonify({"query": query, "encounter_ids": identifiers})

    @app.get("/api/cases/<encounter_id>")
    def api_case(encounter_id: str):
        records = artifact("cases")
        if records is None:
            return api_payload("cases")
        record = records.get(str(encounter_id))
        if record is None:
            return jsonify({"status": "not_found", "encounter_id": encounter_id}), 404
        return jsonify(record)

    @app.get("/api/health")
    def api_health():
        required = tuple(ARTIFACT_FILES)
        missing = missing_artifacts(required)
        manifest = artifact("manifest", {})
        return jsonify(
            {
                "status": "ready" if not missing else "setup_required",
                "ready": not missing,
                "missing_artifacts": missing,
                "manifest": manifest,
                "application": "HealthGraph Readmission Lab",
                "clinical_use": False,
            }
        )

    @app.errorhandler(404)
    def not_found(_: Exception):
        if request.path.startswith("/api/"):
            return jsonify({"status": "not_found", "path": request.path}), 404
        return (
            render_template(
                "setup.html",
                title="Page not found",
                page_id="not-found",
                missing=[],
                command=None,
                message="That dashboard route does not exist.",
            ),
            404,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
