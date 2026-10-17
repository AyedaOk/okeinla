import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rawnind.libs import rawds, rawproc
from rawnind.tools.export_darktable_model import build_model, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rawbench", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("../../models/m8_validation"))
    parser.add_argument("--checkpoint", action="append", metavar="LABEL=PATH", required=True)
    parser.add_argument("--exclusions-from", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.rawbench.resolve() / "src"))
    from rawbench.direct_rgb import demosaic_cfa, recover_model_gain
    from rawbench.rgb_alignment import estimate_rgb_alignment, apply_rgb_alignment
    from rawbench.rgb_metrics import RgbMetricComputer, prepare_rgb_hfen_reference, prepare_rgb_color_reference
    from colour import RGB_COLOURSPACES

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = yaml.safe_load(Path("config/m8_nafnet32_rawforge_detail_seed2026.yaml").read_text())
    dataset = rawds.CleanProfiledRGBNoisyBayerImageCropsValidationDataset(
        config["noise_dataset_yamlfpaths"], 512, config["val_reserve"], True, match_gain="output"
    )
    checkpoints = {}
    for value in args.checkpoint:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in checkpoints:
            parser.error("--checkpoint requires unique LABEL=PATH entries")
        checkpoints[name] = Path(path)
    models = {}
    config["in_channels"] = 4
    for name, path in checkpoints.items():
        model = build_model(config)
        load_checkpoint(model, path)
        models[name] = model.to(args.device).eval()
    conversion = RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB @ RGB_COLOURSPACES["ITU-R BT.2020"].matrix_RGB_to_XYZ

    def render(rgb, matrix):
        rec2020 = rawproc.camRGB_to_lin_rec2020_images(rgb, matrix).squeeze(0).cpu().numpy()
        return np.einsum("ij,jhw->ihw", conversion, rec2020).astype(np.float32)

    args.output.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(args.output / "scores.sqlite")
    database.execute("CREATE TABLE IF NOT EXISTS scores (sample INTEGER, model TEXT, fingerprint TEXT, metrics TEXT, PRIMARY KEY(sample,model))")
    database.execute("CREATE TABLE IF NOT EXISTS exclusions (sample INTEGER PRIMARY KEY, reason TEXT)")
    if args.exclusions_from:
        with sqlite3.connect(f"file:{args.exclusions_from.resolve()}?mode=ro", uri=True) as previous:
            exclusions = previous.execute("SELECT sample,reason FROM exclusions ORDER BY sample").fetchall()
        current = database.execute("SELECT sample,reason FROM exclusions ORDER BY sample").fetchall()
        if current and current != exclusions:
            raise RuntimeError("Exclusions differ; use a new --output directory")
        database.executemany("INSERT OR IGNORE INTO exclusions VALUES (?,?)", exclusions)
        database.commit()
    fingerprint = hashlib.sha256(Path(__file__).read_bytes())
    fingerprint.update(json.dumps(list(checkpoints), ensure_ascii=False).encode())
    if args.exclusions_from:
        fingerprint.update(json.dumps(exclusions).encode())
    for path in checkpoints.values():
        fingerprint.update(hashlib.sha256(path.read_bytes()).digest())
    for path in config["noise_dataset_yamlfpaths"]:
        fingerprint.update(hashlib.sha256(Path(path).read_bytes()).digest())
    fingerprint.update(json.dumps(config, sort_keys=True).encode())
    fingerprint = fingerprint.hexdigest()
    if database.execute("SELECT COUNT(*) FROM scores WHERE fingerprint != ?", (fingerprint,)).fetchone()[0]:
        raise RuntimeError("Inputs or implementation changed; use a new --output directory")
    computer = RgbMetricComputer(args.device)
    with torch.inference_mode():
        for index in range(len(dataset)):
            if database.execute("SELECT 1 FROM exclusions WHERE sample=?", (index,)).fetchone():
                continue
            pending = [name for name in models if not database.execute("SELECT 1 FROM scores WHERE sample=? AND model=?", (index, name)).fetchone()]
            if not pending:
                continue
            batch = dataset[index]
            source = batch["y_crops"].numpy()
            matrix = batch["rgb_xyz_matrix"].unsqueeze(0).to(args.device)
            mask = batch["mask_crops"].numpy()
            target = np.einsum("ij,jhw->ihw", conversion, batch["x_crops"].numpy()).astype(np.float32)
            noisy = render(torch.from_numpy(demosaic_cfa(source)).unsqueeze(0).to(args.device), matrix)
            try:
                alignment = estimate_rgb_alignment(target * mask, noisy * mask)
            except RuntimeError as error:
                if args.exclusions_from:
                    raise RuntimeError(f"Unexpected alignment failure on fixed validation record {index}") from error
                database.execute("INSERT INTO exclusions VALUES (?,?)", (index, str(error)))
                database.commit()
                print(f"Excluded record {index}: {error}", flush=True)
                continue
            gt, _ = apply_rgb_alignment(target * mask, noisy * mask, alignment)
            hfen = prepare_rgb_hfen_reference(gt)
            color = prepare_rgb_color_reference(gt)
            for name in pending:
                output = models[name](batch["y_crops"].unsqueeze(0).to(args.device))
                gain = recover_model_gain(source, output.squeeze(0).cpu().numpy())
                rgb = render(output * gain, matrix)
                _, prediction = apply_rgb_alignment(target * mask, rgb * mask, alignment)
                values = computer.compute(gt, prediction, hfen, color)
                database.execute("INSERT INTO scores VALUES (?,?,?,?)", (index, name, fingerprint, json.dumps(asdict(values))))
                database.commit()
                print(f"{index + 1}/{len(dataset)} {name}: PSNR={values.psnr:.4f} HFEN={values.hfen:.6f} ΔE00={values.ciede2000_mean:.6f}", flush=True)
    excluded = database.execute("SELECT sample,reason FROM exclusions ORDER BY sample").fetchall()
    expected = len(dataset) - len(excluded)
    lines = ["# M8 offline validation", "", f"512-pixel centre crops from {len(dataset)} training-held-out validation records; {expected} scored and {len(excluded)} excluded for all candidates. Shared source-derived alignment/exposure, linear sRGB, masked support. Not comparable to the 133-image DNG benchmark.", "", "| Model | Records | PSNR | HFEN | Mean ΔE00 | MS-SSIM |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in models:
        rows = [json.loads(row[0]) for row in database.execute("SELECT metrics FROM scores WHERE model=?", (name,))]
        assert len(rows) == expected
        means = [np.mean([row[key] for row in rows]) for key in ("psnr", "hfen", "ciede2000_mean", "msssim")]
        lines.append(f"| {name} | {len(rows)} | " + " | ".join(f"{value:.6f}" for value in means) + " |")
    baseline = next(iter(models))
    metric_keys = ("psnr", "hfen", "ciede2000_mean", "msssim")
    scored = {
        name: {index: json.loads(values) for index, values in database.execute("SELECT sample,metrics FROM scores WHERE model=?", (name,))}
        for name in models
    }
    lines.extend(["", f"## Paired changes against {baseline}", "", "Positive PSNR/MS-SSIM and negative HFEN/ΔE00 changes indicate improvement.", "", "| Model | ΔPSNR | ΔHFEN | ΔMean ΔE00 | ΔMS-SSIM |", "| --- | ---: | ---: | ---: | ---: |"])
    for name in models:
        changes = [np.mean([values[key] - scored[baseline][index][key] for index, values in scored[name].items()]) for key in metric_keys]
        lines.append(f"| {name} | " + " | ".join(f"{value:+.6f}" for value in changes) + " |")
    lines.extend(["", "## Per-scene means", "", "| Scene | Model | Records | PSNR | HFEN | Mean ΔE00 | MS-SSIM |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    scenes = sorted({item["image_set"] for item in dataset._dataset})
    for scene in scenes:
        for name in models:
            rows = [values for index, values in scored[name].items() if dataset._dataset[index]["image_set"] == scene]
            if rows:
                means = [np.mean([row[key] for row in rows]) for key in metric_keys]
                lines.append(f"| {scene} | {name} | {len(rows)} | " + " | ".join(f"{value:.6f}" for value in means) + " |")
    lines.extend(["", "## Shared exclusions", ""] + [f"- Record {index}: {reason}" for index, reason in excluded])
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    database.close()


if __name__ == "__main__":
    main()
