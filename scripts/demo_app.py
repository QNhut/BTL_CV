"""Demo web (Gradio): tải ảnh cây lên, chọn model, xem số quả đếm được.

    python scripts/demo_app.py --models configs/models/yolov8.yaml configs/models/csrnet.yaml
"""
import argparse

from _common import ROOT, default_weights

from src.inference import CountingPredictor
from src.visualization import render_result


def main() -> None:
    import gradio as gr

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="các file config model đã huấn luyện")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    predictors: dict[str, CountingPredictor] = {}
    for cfg_path in args.models:
        p = CountingPredictor.from_config(cfg_path)
        p.model.load(str(default_weights(args.runs, p.model.cfg)))
        predictors[p.model.cfg["name"]] = p

    def run(image, model_name, threshold, tile):
        if image is None:
            return None, "Hãy tải một ảnh lên."
        p = predictors[model_name]
        if p.model.family != "density":
            p.set_count_threshold(threshold)
            p.set_tiling(tile)
        res = p.predict_array(image)
        return render_result(image, res), f"Số quả đếm được: {int(round(res.count))}"

    gr.Interface(
        fn=run,
        inputs=[
            gr.Image(type="numpy", label="Ảnh cây"),
            gr.Dropdown(list(predictors), value=list(predictors)[0], label="Model"),
            gr.Slider(0.05, 0.95, value=0.3, step=0.05, label="Ngưỡng đếm (model phát hiện)"),
            gr.Checkbox(value=False, label="Tiling (quả nhỏ / ảnh lớn)"),
        ],
        outputs=[gr.Image(label="Kết quả"), gr.Textbox(label="Số quả")],
        title="FruitCount — đếm trái cây trên cây",
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
