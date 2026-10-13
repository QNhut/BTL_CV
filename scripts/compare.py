"""Gộp kết quả các model thành bảng so sánh + biểu đồ (dùng cho báo cáo).

    python scripts/compare.py --split test
"""
import argparse
import logging
from pathlib import Path

from _common import ROOT, setup_logging

from src.evaluation import write_csv
from src.utils import read_json
from src.visualization import plot_error_by_bin, plot_model_comparison

log = logging.getLogger("fruitcount")

COLUMNS = ["model", "family", "mae", "rmse", "mape_pct", "r2", "map50", "map", "fps", "params_million"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None, help="mặc định: <runs>/comparison")
    args = ap.parse_args()
    setup_logging()

    runs = Path(args.runs)
    out = Path(args.out) if args.out else runs / "comparison"
    rows, metrics, bins = [], {}, {}
    for f in sorted(runs.glob(f"*/metrics_{args.split}*.json")):
        r = read_json(f)
        name = r["model"] + (" (tile)" if r.get("tiling") else "")
        c, d, s = r["counting"], r.get("detection", {}), r.get("speed", {})
        row = {
            "model": name, "family": r["family"],
            "mae": c["mae"], "rmse": c["rmse"], "mape_pct": c["mape_pct"], "r2": c["r2"],
            "map50": d.get("map50", ""), "map": d.get("map", ""),
            "fps": s.get("fps", ""), "params_million": s.get("params_million", ""),
        }
        rows.append(row)
        metrics[name] = {k: v for k, v in row.items() if isinstance(v, float)}
        bins[name] = r["error_by_count_bin"]
    if not rows:
        raise SystemExit(f"Chưa có metrics_{args.split}*.json trong {runs}; hãy chạy scripts/evaluate.py trước.")

    rows.sort(key=lambda r: r["mae"])
    write_csv([{k: r[k] for k in COLUMNS} for r in rows], out / f"comparison_{args.split}.csv")
    plot_model_comparison(metrics, ("mae", "rmse", "mape_pct"), out / f"comparison_{args.split}.png")
    plot_error_by_bin(bins, out / f"error_by_count_{args.split}.png")

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else (v if v != "" else "-")

    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(fmt(r[k]) for k in COLUMNS) + " |" for r in rows]
    (out / f"comparison_{args.split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n" + "\n".join(lines))
    log.info("Đã ghi bảng và biểu đồ vào %s", out)


if __name__ == "__main__":
    main()
