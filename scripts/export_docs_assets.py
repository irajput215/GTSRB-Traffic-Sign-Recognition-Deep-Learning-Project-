#!/usr/bin/env python
"""Generate the architecture diagrams used in the README and docs.

The diagrams are produced by code rather than drawn by hand so that they can be
regenerated when the architecture changes, and so a reviewer can see exactly what
each one claims. Every diagram is a PNG under ``docs/images/``.

The same flows are also written as Mermaid in ``docs/ARCHITECTURE.md``. Mermaid is
maintainable and diffable; the rendered PNGs are what make the README readable on
GitHub without a Mermaid renderer. Both are kept, deliberately, because they serve
different readers.

Usage:
    python scripts/export_docs_assets.py [--output-dir docs/images]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "images"

DPI = 160
FONT = "DejaVu Sans"

# Muted, distinguishable palette. Chosen so the diagrams survive being printed in
# greyscale and so no colour carries meaning on its own - every box is also
# labelled.
PALETTE: dict[str, tuple[str, str]] = {
    "data": ("#E8F1F8", "#2E6DA4"),
    "process": ("#EAF3EA", "#3C8C40"),
    "artifact": ("#FDF3E3", "#C77A16"),
    "service": ("#F0EAF6", "#6B4C9A"),
    "deploy": ("#FBEAEA", "#B03A3A"),
    "external": ("#F2F2F2", "#666666"),
    "observability": ("#E9F5F5", "#2A8C8C"),
}

TEXT = "#1A1A1A"


@dataclass
class Node:
    """One box in a flow diagram."""

    title: str
    subtitle: str | None = None
    kind: str = "process"
    side: list[Node] = field(default_factory=list)


def _figure(rows: int, width: float = 8.4, row_height: float = 0.95) -> tuple[plt.Figure, plt.Axes]:
    height = rows * row_height + 0.9
    figure, axes = plt.subplots(figsize=(width, height))
    axes.set_xlim(0, 10)
    axes.set_ylim(0, rows * row_height + 0.9)
    axes.axis("off")
    return figure, axes


def _box(
    axes: plt.Axes,
    node: Node,
    bounds: tuple[float, float, float, float],
    *,
    fontsize: float = 10.0,
) -> None:
    """Draw ``node`` inside ``bounds`` = ``(x, y, width, height)``."""
    x, y, width, height = bounds
    fill, edge = PALETTE[node.kind]
    patch = mpatches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.09",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=fill,
        zorder=2,
    )
    axes.add_patch(patch)

    if node.subtitle:
        axes.text(
            x + width / 2,
            y + height * 0.63,
            node.title,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=TEXT,
            family=FONT,
            zorder=3,
        )
        axes.text(
            x + width / 2,
            y + height * 0.27,
            node.subtitle,
            ha="center",
            va="center",
            fontsize=fontsize - 2.4,
            color="#4A4A4A",
            family=FONT,
            zorder=3,
        )
    else:
        axes.text(
            x + width / 2,
            y + height / 2,
            node.title,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=TEXT,
            family=FONT,
            zorder=3,
        )


def _arrow(
    axes: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str | None = None,
    color: str = "#5A5A5A",
    style: str = "-",
    dashed: bool = False,
    label_offset: float = 0.11,
) -> None:
    arrow = mpatches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.3,
        color=color,
        linestyle="--" if dashed else style,
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    )
    axes.add_patch(arrow)
    if label:
        axes.text(
            (start[0] + end[0]) / 2 + label_offset,
            (start[1] + end[1]) / 2,
            label,
            ha="left",
            va="center",
            fontsize=7.6,
            color=color,
            family=FONT,
            zorder=3,
            path_effects=[withStroke(linewidth=2.6, foreground="white")],
        )


def _draw_flow(
    nodes: list[Node],
    output: Path,
    *,
    title: str,
    footnote: str | None = None,
    box_width: float = 5.4,
    side_width: float = 3.4,
    row_height: float = 0.95,
) -> Path:
    """Draw a vertical flow, with side boxes stacked beside their parent.

    Layout is computed in *units* rather than pixels: a node with ``k`` side boxes
    occupies ``k`` units of vertical space, and its main box spans all of them. That
    is what keeps several side boxes attached to one stage from landing on top of
    each other, which is exactly what happens if each node is assumed to occupy one
    row.
    """
    units_per_node = [max(1, len(node.side)) for node in nodes]
    total_units = sum(units_per_node)
    # Extra room below the flow for the footnote, so it cannot overlap the last box.
    footnote_space = 1.0 if footnote else 0.25
    canvas_height = total_units * row_height + footnote_space

    figure, axes = plt.subplots(figsize=(8.6, canvas_height / 1.35))
    axes.set_xlim(0, 10)
    axes.set_ylim(0, canvas_height)
    axes.axis("off")

    left = 0.5
    centre_x = left + box_width / 2
    side_left = left + box_width + 0.6
    inset = row_height * 0.14
    offset = 0

    for index, node in enumerate(nodes):
        span = units_per_node[index] * row_height
        # Everything is shifted up by footnote_space so the footnote sits below the
        # flow rather than inside the last box.
        top = footnote_space + (total_units - offset) * row_height
        bottom = top - span
        offset += units_per_node[index]

        _box(axes, node, (left, bottom + inset, box_width, span - 2 * inset))

        if index < len(nodes) - 1:
            _arrow(axes, (centre_x, bottom + inset), (centre_x, bottom - inset))

        for side_index, side_node in enumerate(node.side):
            side_top = top - side_index * row_height
            side_bottom = side_top - row_height
            _box(
                axes,
                side_node,
                (side_left, side_bottom + inset, side_width, row_height - 2 * inset),
                fontsize=8.8,
            )
            _arrow(
                axes,
                (left + box_width, side_bottom + row_height / 2),
                (side_left, side_bottom + row_height / 2),
                dashed=True,
                color=PALETTE[side_node.kind][1],
            )

    axes.set_title(title, fontsize=13.5, fontweight="bold", color=TEXT, family=FONT, pad=16)
    if footnote:
        axes.text(
            5.0,
            footnote_space * 0.42,
            footnote,
            ha="center",
            va="center",
            fontsize=7.4,
            color="#5A5A5A",
            family=FONT,
        )
    return _save(figure, output)


def _save(figure: plt.Figure, output: Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"wrote {output}")
    return output


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------
def system_architecture(output: Path) -> Path:
    """Training-time and inference-time halves of the system, and where they meet."""
    nodes = [
        Node("GTSRB dataset", "39,209 official · 26,640 via torchvision · 43 classes", "external"),
        Node(
            "Validation", "label range · image mode · aspect ratio · class distribution", "process"
        ),
        Node(
            "Split manifest",
            "stratified 80/20, seed 43, indices + labels persisted and verified",
            "artifact",
        ),
        Node(
            "Preprocessing and augmentation",
            "resize · rotate · translate · colour jitter · normalise (train only)",
            "process",
        ),
        Node(
            "Training pipeline",
            "configurable hyperparameters · AMP · grad clip · early stopping · resume",
            "process",
            side=[
                Node("MLflow", "params · metrics · config artifact", "observability"),
                Node("Checkpoint", "weights + config + labels + metrics", "artifact"),
            ],
        ),
        Node(
            "Evaluation",
            "accuracy · macro F1 · top-k · per-class · per-family · ECE",
            "process",
            side=[
                Node("Error analysis", "worst classes · confusions · calibration", "observability"),
                Node("Artifacts", "metrics · CSV · figures · report", "artifact"),
            ],
        ),
        Node("Inference predictor", "validated preprocessing · single and batch", "service"),
        Node("FastAPI service", "/predict · /health · /ready · /metrics", "service"),
        Node("Docker image", "1.8 GB · CPU PyTorch · non-root · healthcheck", "deploy"),
        Node("Client", "HTTP or Swagger UI", "external"),
    ]
    return _draw_flow(
        nodes,
        output / "system_architecture.png",
        title="GTSRB traffic sign recognition — system architecture",
        footnote=(
            "Training half: dataset to checkpoint.   "
            "Inference half: checkpoint to client.   "
            "Dashed boxes are side outputs, not pipeline stages."
        ),
        row_height=0.98,
    )


def training_pipeline(output: Path) -> Path:
    nodes = [
        Node("Resolved configuration", "Pydantic v2 · defaults < YAML < env < CLI", "artifact"),
        Node("Seed everything", "python · numpy · torch · cuDNN · DataLoader workers", "process"),
        Node("DataLoaders", "train (augmented, balanced) · val (clean) · test (clean)", "data"),
        Node("Model factory", "compact_cnn · mlp · resnet50", "process"),
        Node(
            "Per epoch",
            "forward · loss · backward · clip · step · validate · schedule",
            "process",
            side=[
                Node("AMP", "torch.amp, CUDA only", "observability"),
                Node("ReduceLROnPlateau", "on the monitored metric", "observability"),
            ],
        ),
        Node("Metric check", "early stopping on val_macro_f1 · best checkpoint", "process"),
        Node("Checkpoint", "best.pt (weights) + last.pt (resumable)", "artifact"),
        Node("Run summary", "train_result.json + run log", "artifact"),
    ]
    return _draw_flow(
        nodes,
        output / "training_pipeline.png",
        title="Training pipeline",
        footnote=(
            "Leakage discipline: split first, then balance and augment the training partition only."
        ),
    )


def inference_pipeline(output: Path) -> Path:
    nodes = [
        Node("HTTP upload", "multipart form field 'file'", "external"),
        Node("Size check", "before decoding, by byte length", "process"),
        Node(
            "Decode and validate", "Pillow · mode to RGB · bomb limit · truncated files", "process"
        ),
        Node(
            "Checkpoint preprocessing", "image size + mean/std read from the checkpoint", "process"
        ),
        Node("Model forward", "inference_mode · eval · single or batched", "service"),
        Node("Softmax", "probabilities over 43 classes", "service"),
        Node("Ranked class scores", "class id · name · family · confidence", "artifact"),
        Node("JSON response", "top_predictions + latency + model version", "external"),
    ]
    return _draw_flow(
        nodes,
        output / "inference_pipeline.png",
        title="Inference pipeline",
        footnote=(
            "Validation decodes the payload rather than trusting a filename or Content-Type. "
            "Uploads are never written to disk or logged."
        ),
    )


def ml_lifecycle(output: Path) -> Path:
    nodes = [
        Node("Configure", "YAML layers + CLI overrides", "artifact"),
        Node("Experiment", "MLflow run: params, resolved config artifact", "observability"),
        Node("Train", "epoch metrics streamed to the run", "process"),
        Node("Evaluate", "test-split metrics + error analysis", "process"),
        Node("Select", "best val_macro_f1 checkpoint", "artifact"),
        Node(
            "Register",
            "tracking.register_model=true (opt-in)",
            "observability",
            side=[Node("Model registry", "versioned, loadable by URI", "observability")],
        ),
        Node("Serve", "FastAPI + Docker from the same checkpoint", "service"),
        Node("Monitor", "latency · error rate · predicted-class distribution", "observability"),
        Node("Retrain", "on drift signal or new data", "process"),
    ]
    return _draw_flow(
        nodes,
        output / "ml_lifecycle.png",
        title="Model lifecycle",
        footnote=(
            "Registering is opt-in: a registry entry is a claim that a model is worth promoting."
        ),
    )


def ci_cd_pipeline(output: Path) -> Path:
    nodes = [
        Node("Pull request", "feature branch", "external"),
        Node("lint", "ruff check + format check + config validation · seconds", "process"),
        Node("typecheck", "mypy with disallow_untyped_defs", "process"),
        Node("test", "pytest on Python 3.12 and 3.13 · coverage + JUnit", "process"),
        Node(
            "docker",
            "build image · assert /health 200 and /ready 503 without a checkpoint",
            "deploy",
        ),
        Node("ci-complete", "single required check for branch protection", "artifact"),
        Node("Merge to main", "squashed feature history preserved", "external"),
        Node(
            "model-validation",
            "on demand + weekly: download, train, gate, evaluate, serve",
            "observability",
        ),
    ]
    return _draw_flow(
        nodes,
        output / "ci_cd_pipeline.png",
        title="CI/CD pipeline",
        footnote="Model validation is deliberately not on every PR: training is minutes to hours.",
        row_height=0.92,
    )


def repository_structure(output: Path) -> Path:
    """A grouped view of the tree, with the purpose of each group."""
    groups: list[tuple[str, str, list[str], str]] = [
        (
            "src/gtsrb/",
            "the installable package",
            [
                "config/      validated configuration, label space",
                "data/        dataset, splits, preprocessing, augmentation",
                "models/      architectures, factory, checkpoints",
                "training/    trainer, losses, metrics, callbacks",
                "tracking/    MLflow tracker (optional extra)",
                "evaluation/  metrics, error analysis, figures",
                "inference/   predictor, input validation",
                "api/         FastAPI service",
                "cli/         train, evaluate, predict, serve",
                "runtime.py   device, seeding, logging",
            ],
            "process",
        ),
        (
            "tests/",
            "unit and integration suites",
            [
                "unit/          fast, no dataset, synthetic fixtures",
                "integration/   real wiring; skips cleanly without the dataset",
                "conftest.py    shared synthetic fixtures",
            ],
            "data",
        ),
        (
            "configs/",
            "externalised configuration",
            [
                "data.yaml   dataset, geometry, normalisation, augmentation",
                "model.yaml  architecture selection and settings",
                "train.yaml  optimiser, schedule, stopping, checkpointing",
                "api.yaml    service settings",
            ],
            "artifact",
        ),
        (
            "docs/",
            "the written record",
            [
                "PROJECT_AUDIT.md    what existed and what was wrong",
                "ARCHITECTURE.md     how the system works",
                "DECISIONS.md        why, with alternatives and trade-offs",
                "EXPERIMENT_TRACKING.md  MLflow usage and trade-offs",
                "DEPLOYMENT.md           containers, scaling, limitations",
                "images/             generated diagrams",
            ],
            "observability",
        ),
        (
            ".github/",
            "automation",
            [
                "workflows/ci.yml               lint, typecheck, test, docker",
                "workflows/model-validation.yml   scheduled training gate",
            ],
            "deploy",
        ),
        (
            "scripts/",
            "entry points that are not part of the package API",
            [
                "prepare_data.py         download, validate, write the split",
                "validate_configs.py     config gate for CI",
                "check_model_quality.py  model quality gate for CI",
                "export_docs_assets.py   generates these diagrams",
            ],
            "external",
        ),
    ]

    # Geometry derived from content, so a group with ten entries cannot overflow
    # its box the way a fixed height would.
    line_height = 0.30
    header_height = 1.05
    padding = 0.30
    columns = 2
    column_width = 9.5
    gap = 0.5

    heights = [header_height + len(entries) * line_height + padding for _, _, entries, _ in groups]
    rows = (len(groups) + columns - 1) // columns
    row_heights = [max(heights[row * columns : (row + 1) * columns]) + gap for row in range(rows)]
    total_height = sum(row_heights)

    figure, axes = plt.subplots(figsize=(13.6, total_height / 1.15))
    axes.set_xlim(0, columns * column_width + gap)
    axes.set_ylim(0, total_height)
    axes.axis("off")

    for index, (name, purpose, entries, kind) in enumerate(groups):
        column = index % columns
        row = index // columns
        box_height = heights[index]
        x = 0.3 + column * (column_width + gap)
        # Top-aligned within the row so boxes of different heights line up cleanly.
        y = total_height - sum(row_heights[: row + 1]) + (row_heights[row] - gap - box_height)

        fill, edge = PALETTE[kind]
        axes.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                column_width - 0.2,
                box_height,
                boxstyle="round,pad=0.03,rounding_size=0.10",
                linewidth=1.5,
                edgecolor=edge,
                facecolor=fill,
                zorder=2,
            )
        )
        axes.text(
            x + 0.30,
            y + box_height - 0.34,
            name,
            fontsize=11.5,
            fontweight="bold",
            color=TEXT,
            family=FONT,
            va="center",
            zorder=3,
        )
        axes.text(
            x + 0.30,
            y + box_height - 0.68,
            purpose,
            fontsize=8.6,
            color="#4A4A4A",
            family=FONT,
            va="center",
            style="italic",
            zorder=3,
        )
        for line_index, entry in enumerate(entries):
            axes.text(
                x + 0.36,
                y + box_height - 1.05 - line_index * line_height,
                entry,
                fontsize=8.4,
                color=TEXT,
                family="DejaVu Sans Mono",
                va="center",
                zorder=3,
            )

    axes.set_title(
        "Repository structure — grouped by responsibility",
        fontsize=15,
        fontweight="bold",
        color=TEXT,
        family=FONT,
        pad=18,
    )
    return _save(figure, output / "repository_structure.png")


def testing_flow(output: Path) -> Path:
    nodes = [
        Node("Synthetic fixtures", "sign-like images + tiny models · no dataset needed", "data"),
        Node(
            "Unit tests",
            "config · data · models · training · evaluation · inference · scripts",
            "process",
        ),
        Node(
            "Integration tests", "real pipeline · real MLflow store · real FastAPI app", "process"
        ),
        Node("Dataset-dependent tests", "skip cleanly when GTSRB is absent", "data"),
        Node("Quality gate", "ruff · mypy --strict-ish · pytest", "process"),
        Node("Coverage and JUnit", "uploaded as CI artifacts", "artifact"),
    ]
    return _draw_flow(
        nodes,
        output / "testing_flow.png",
        title="Testing flow",
        footnote="A test suite that needs a 300 MB download is a test suite nobody runs.",
    )


DIAGRAMS: dict[str, object] = {
    "system_architecture": system_architecture,
    "training_pipeline": training_pipeline,
    "inference_pipeline": inference_pipeline,
    "ml_lifecycle": ml_lifecycle,
    "ci_cd_pipeline": ci_cd_pipeline,
    "repository_structure": repository_structure,
    "testing_flow": testing_flow,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(DIAGRAMS),
        help="regenerate only these diagrams",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected = args.only or sorted(DIAGRAMS)
    for name in selected:
        DIAGRAMS[name](args.output_dir)  # type: ignore[operator]
    print(f"\n{len(selected)} diagram(s) written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
