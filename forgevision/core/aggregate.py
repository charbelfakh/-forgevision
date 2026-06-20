"""
Aggregate per-category results and head-to-head method comparison.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from forgevision.core.utils import ensure_dir

CANONICAL_CATEGORIES: tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


def discover_present_categories(data_root: Path) -> list[str]:
    """Find categories whose folder contains train/good/."""
    if not data_root.is_dir():
        return []
    present: list[str] = []
    for path in sorted(data_root.iterdir()):
        if path.is_dir() and (path / "train" / "good").is_dir():
            present.append(path.name)
    return present


def resolve_categories(
    data_root: Path,
    subset: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return (to_run, present, missing)."""
    present = discover_present_categories(data_root)
    missing = [c for c in CANONICAL_CATEGORIES if c not in present]

    if subset:
        unknown = [c for c in subset if c not in present]
        if unknown:
            raise ValueError(
                f"Requested categories not found under {data_root}: {unknown}\n"
                f"Present: {present}"
            )
        to_run = subset
    else:
        to_run = present

    return to_run, present, missing


def _sort_results(results: list[dict]) -> list[dict]:
    def sort_key(row: dict) -> tuple[int, float]:
        img = row.get("image_auroc", float("nan"))
        if img != img:
            return (1, 0.0)
        return (0, -img)

    return sorted(results, key=sort_key)


def _nanmean(values: list[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _fmt(value: float) -> str:
    return f"{value:.3f}" if value == value else "NaN"


def write_results_md(results: list[dict], output_path: Path, title: str = "") -> None:
    """Write markdown table sorted by image AUROC."""
    ensure_dir(output_path.parent)
    sorted_rows = _sort_results(results)
    mean_img = _nanmean([r["image_auroc"] for r in results])
    mean_px = _nanmean([r["pixel_auroc"] for r in results])

    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("")
    lines.extend(
        [
            "| Category | Image AUROC | Pixel AUROC |",
            "|----------|-------------|-------------|",
        ]
    )
    for row in sorted_rows:
        lines.append(
            f"| {row['category']} | {_fmt(row['image_auroc'])} | {_fmt(row['pixel_auroc'])} |"
        )
    lines.append(f"| **Mean** | **{_fmt(mean_img)}** | **{_fmt(mean_px)}** |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results_csv(results: list[dict], output_path: Path) -> None:
    """Write machine-readable CSV (one row per category + mean row)."""
    ensure_dir(output_path.parent)
    sorted_rows = _sort_results(results)
    mean_img = _nanmean([r["image_auroc"] for r in results])
    mean_px = _nanmean([r["pixel_auroc"] for r in results])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "image_auroc", "pixel_auroc", "error"],
        )
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(
                {
                    "category": row["category"],
                    "image_auroc": row.get("image_auroc", float("nan")),
                    "pixel_auroc": row.get("pixel_auroc", float("nan")),
                    "error": row.get("error", ""),
                }
            )
        writer.writerow(
            {
                "category": "mean",
                "image_auroc": mean_img,
                "pixel_auroc": mean_px,
                "error": "",
            }
        )


def write_results_plot(
    results: list[dict],
    output_path: Path,
    title: str = "ForgeVision baseline (conv autoencoder) — image AUROC by category",
) -> None:
    """Horizontal bar chart of image AUROC per category."""
    ensure_dir(output_path.parent)
    valid = [r for r in results if r.get("image_auroc") == r.get("image_auroc")]
    if not valid:
        print("Warning: no valid image AUROC scores — skipping plot.")
        return

    sorted_rows = _sort_results(valid)
    categories = [r["category"] for r in sorted_rows]
    scores = [r["image_auroc"] for r in sorted_rows]
    mean_score = _nanmean(scores)

    fig, ax = plt.subplots(figsize=(10, max(5, len(categories) * 0.35)))
    y_pos = np.arange(len(categories))
    ax.barh(y_pos, scores, color="#4C72B0", edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_xlabel("Image AUROC")
    ax.set_xlim(0, 1.0)
    ax.axvline(
        mean_score,
        color="#C44E52",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {mean_score:.3f}",
    )
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def load_results_csv(path: Path) -> dict[str, dict]:
    """Load results CSV into {category: {image_auroc, pixel_auroc}} (excludes mean row)."""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = row["category"]
            if cat == "mean":
                continue
            out[cat] = {
                "image_auroc": float(row["image_auroc"]) if row["image_auroc"] else float("nan"),
                "pixel_auroc": float(row["pixel_auroc"]) if row["pixel_auroc"] else float("nan"),
            }
    return out


def write_comparison_md(
    ae: dict[str, dict],
    pc: dict[str, dict],
    output_path: Path,
) -> None:
    """Head-to-head markdown table sorted by Δimage descending."""
    ensure_dir(output_path.parent)
    categories = sorted(set(ae) | set(pc))
    rows: list[dict] = []
    for cat in categories:
        ae_img = ae.get(cat, {}).get("image_auroc", float("nan"))
        pc_img = pc.get(cat, {}).get("image_auroc", float("nan"))
        ae_px = ae.get(cat, {}).get("pixel_auroc", float("nan"))
        pc_px = pc.get(cat, {}).get("pixel_auroc", float("nan"))
        d_img = pc_img - ae_img if (pc_img == pc_img and ae_img == ae_img) else float("nan")
        d_px = pc_px - ae_px if (pc_px == pc_px and ae_px == ae_px) else float("nan")
        rows.append(
            {
                "category": cat,
                "ae_img": ae_img,
                "pc_img": pc_img,
                "d_img": d_img,
                "ae_px": ae_px,
                "pc_px": pc_px,
                "d_px": d_px,
            }
        )

    rows.sort(
        key=lambda r: (r["d_img"] != r["d_img"], -(r["d_img"] if r["d_img"] == r["d_img"] else 0)),
    )

    lines = [
        "| Category | AE image | PatchCore image | Δimage | AE pixel | PatchCore pixel | Δpixel |",
        "|----------|----------|-----------------|--------|----------|-----------------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['category']} | {_fmt(r['ae_img'])} | {_fmt(r['pc_img'])} | "
            f"{_fmt(r['d_img'])} | {_fmt(r['ae_px'])} | {_fmt(r['pc_px'])} | {_fmt(r['d_px'])} |"
        )

    mean_ae_img = _nanmean([r["ae_img"] for r in rows])
    mean_pc_img = _nanmean([r["pc_img"] for r in rows])
    mean_d_img = mean_pc_img - mean_ae_img if (mean_pc_img == mean_pc_img and mean_ae_img == mean_ae_img) else float("nan")
    mean_ae_px = _nanmean([r["ae_px"] for r in rows])
    mean_pc_px = _nanmean([r["pc_px"] for r in rows])
    mean_d_px = mean_pc_px - mean_ae_px if (mean_pc_px == mean_pc_px and mean_ae_px == mean_ae_px) else float("nan")
    lines.append(
        f"| **Mean** | **{_fmt(mean_ae_img)}** | **{_fmt(mean_pc_img)}** | "
        f"**{_fmt(mean_d_img)}** | **{_fmt(mean_ae_px)}** | **{_fmt(mean_pc_px)}** | "
        f"**{_fmt(mean_d_px)}** |"
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_plot(
    ae: dict[str, dict],
    pc: dict[str, dict],
    output_path: Path,
) -> None:
    """Grouped bar chart: AE vs PatchCore image AUROC per category."""
    ensure_dir(output_path.parent)
    categories = sorted(set(ae) & set(pc))
    if not categories:
        print("Warning: no overlapping categories for comparison plot.")
        return

    ae_scores = [ae[c]["image_auroc"] for c in categories]
    pc_scores = [pc[c]["image_auroc"] for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, ae_scores, width, label="Autoencoder", color="#4C72B0")
    ax.bar(x + width / 2, pc_scores, width, label="PatchCore", color="#55A868")
    ax.set_ylabel("Image AUROC")
    ax.set_xlabel("Category")
    ax.set_title("ForgeVision — Autoencoder vs PatchCore (image AUROC)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def aggregate_and_save(results: list[dict], eval_dir: Path, method: str) -> None:
    """Write method-specific summary artefacts under eval/."""
    if method == "autoencoder":
        write_results_md(results, eval_dir / "results.md")
        write_results_csv(results, eval_dir / "results.csv")
        write_results_plot(
            results,
            eval_dir / "results_plot.png",
            title="ForgeVision baseline (conv autoencoder) — image AUROC by category",
        )
    elif method == "patchcore":
        write_results_csv(results, eval_dir / "results_patchcore.csv")
        ae_path = eval_dir / "results.csv"
        if ae_path.exists():
            ae = load_results_csv(ae_path)
            pc = load_results_csv(eval_dir / "results_patchcore.csv")
            write_comparison_md(ae, pc, eval_dir / "comparison.md")
            write_comparison_plot(ae, pc, eval_dir / "comparison_plot.png")
        else:
            print("Note: eval/results.csv not found — skipping comparison artefacts.")


def aggregate_comparison(eval_dir: Path) -> None:
    """Regenerate comparison.md/plot from existing AE + PatchCore CSVs."""
    ae_path = eval_dir / "results.csv"
    pc_path = eval_dir / "results_patchcore.csv"
    if not ae_path.exists() or not pc_path.exists():
        print("Cannot build comparison — need both results.csv and results_patchcore.csv.")
        return
    ae = load_results_csv(ae_path)
    pc = load_results_csv(pc_path)
    write_comparison_md(ae, pc, eval_dir / "comparison.md")
    write_comparison_plot(ae, pc, eval_dir / "comparison_plot.png")
