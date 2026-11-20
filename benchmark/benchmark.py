"""Download and score Okeinla/RawNIND Bayer models; see benchmark.md.

Benchmark implementation by Ayeda Okambawa.
This is a new direct-RGB protocol, not the historical darktable DNG benchmark.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
import urllib.parse
import urllib.request

import cv2
import numpy as np

from benchmark_alignment import apply_rgb_alignment, estimate_rgb_alignment


DATASET = "AyedaOk/rawbench-dataset"
REVISION = "978c222c700b2dda9b4aa96af5fc54dad1fa6e87"
PROTOCOL = "darktable-ai-direct-rgb-v1"
# Scene 13/15 are explicit new selections, not verified historical identities.
REFERENCES = (2, 3, 2, 2, 1, 3, 2, 3, 3, 1, 1, 1, 1, 3, 3)
TILE = 512
STEP = 448
XYZ_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660, 1.8760108, 0.0415560],
                        [0.0556434, -0.2040259, 1.0572252]])


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, path, expected):
    """Check cached files too; publish downloads only after hash verification."""
    if path.is_file() and sha256(path) == expected:
        return path
    if path.exists():
        raise RuntimeError(f"cached file failed SHA256 verification: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        if sha256(partial) != expected:
            raise RuntimeError(f"download failed SHA256 verification: {path}")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def dataset_manifest():
    url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}?recursive=true&limit=1000"
    with urllib.request.urlopen(url, timeout=60) as response:
        if response.headers.get("Link"):
            raise RuntimeError("dataset listing is paginated; update the pinned manifest loader")
        entries = json.load(response)
    files = {}
    for entry in entries:
        name = entry["path"]
        if not name.endswith(".ARW"):
            continue
        if not re.fullmatch(r"Scene_\d{4}/(?:gt/)?Scene_\d{4}_ISO\d+_\d+\.ARW", name):
            raise RuntimeError(f"unexpected dataset path: {name}")
        digest = entry.get("lfs", {}).get("oid", "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"missing SHA256 for {name}")
        files[name] = (digest, entry["size"])
    sources = sorted(name for name in files if "/gt/" not in name)
    if len(sources) != 133 or len(files) != 177:
        raise RuntimeError("pinned dataset does not contain the expected 133 noisy/44 reference files")
    pairs = []
    for source in sources:
        scene = source.split("/")[0]
        index = int(scene.removeprefix("Scene_"))
        if not 1 <= index <= len(REFERENCES):
            raise RuntimeError(f"unexpected scene: {scene}")
        reference = f"{scene}/gt/{scene}_ISO100_{REFERENCES[index - 1]}.ARW"
        if reference not in files:
            raise RuntimeError(f"selected reference is missing: {reference}")
        pairs.append((source, reference))
    return files, pairs


def decode(path):
    import rawpy

    with rawpy.imread(str(path)) as raw:
        pattern = raw.raw_pattern
        if pattern is None or pattern.shape != (2, 2):
            raise RuntimeError(f"not a Bayer RAW: {path}")
        colors = tuple(chr(raw.color_desc[int(v)]) for v in pattern.flat)
        if colors != ("R", "G", "G", "B"):
            raise RuntimeError(f"benchmark's Sony crop requires RGGB: {path}")
        visible = raw.raw_image_visible
        if visible.shape[0] < 4680 or visible.shape[1] < 7020:
            raise RuntimeError(f"RAW does not cover the fixed benchmark crop: {path}")
        crop = visible[8:4680, 12:7020].astype(np.float32)
        planes = []
        for y, x in ((0, 0), (0, 1), (1, 0), (1, 1)):
            black = raw.black_level_per_channel[int(pattern[y, x])]
            if raw.white_level <= black:
                raise RuntimeError(f"invalid black/white levels: {path}")
            planes.append((crop[y::2, x::2] - black) / (raw.white_level - black))
        profile = (np.array(raw.camera_whitebalance, dtype=np.float64),
                   np.array(raw.rgb_xyz_matrix[:3, :3], dtype=np.float64))
    packed = np.ascontiguousarray(planes, dtype=np.float32)
    if not np.isfinite(packed).all():
        raise RuntimeError(f"non-finite RAW samples: {path}")
    return packed, profile


def demosaic(packed):
    mosaic = np.empty((packed.shape[1] * 2, packed.shape[2] * 2), dtype=np.float32)
    for channel, (y, x) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        mosaic[y::2, x::2] = packed[channel]
    rgb = np.zeros((3, *mosaic.shape), dtype=np.float32)
    rgb[0, 0::2, 0::2] = mosaic[0::2, 0::2]
    rgb[1, 0::2, 1::2] = mosaic[0::2, 1::2]
    rgb[1, 1::2, 0::2] = mosaic[1::2, 0::2]
    rgb[2, 1::2, 1::2] = mosaic[1::2, 1::2]
    rb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 4
    green = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float32) / 4
    return np.stack([cv2.filter2D(rgb[c], -1, green if c == 1 else rb,
                                 borderType=cv2.BORDER_REFLECT_101) for c in range(3)])


def convert_color(rgb, profile):
    white_balance, matrix = profile
    wb = np.array([white_balance[0], (white_balance[1] + white_balance[3]) / 2,
                   white_balance[2]])
    if not np.isfinite(wb).all() or (wb <= 0).any():
        raise RuntimeError("invalid reference white balance")
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-10:
        raise RuntimeError("invalid reference camera matrix")
    srgb_to_camera = matrix @ np.linalg.inv(XYZ_TO_SRGB)
    white_response = srgb_to_camera.sum(axis=1)
    if not np.isfinite(white_response).all() or (white_response <= 0).any():
        raise RuntimeError("invalid camera response to D65 white")
    transform = np.linalg.inv(srgb_to_camera / white_response[:, None])
    balanced = rgb * (wb / wb[1])[:, None, None]
    return np.ascontiguousarray(np.einsum("ij,jhw->ihw", transform, balanced), dtype=np.float32)


def recover_gain(source, rgb):
    cfa = np.stack((rgb[0, 0::2, 0::2], rgb[1, 0::2, 1::2],
                    rgb[1, 1::2, 0::2], rgb[2, 1::2, 1::2])).astype(np.float64)
    valid = (source > 0.001) & (source < 0.999)
    if not np.isfinite(rgb).all() or valid.mean() < 0.1:
        raise RuntimeError("invalid output or insufficient source samples for gain recovery")
    values = cfa[valid]
    denominator = np.dot(values, values)
    if denominator <= 1e-20:
        raise RuntimeError("model output has insufficient energy")
    gain = float(np.dot(values, source[valid].astype(np.float64)) / denominator)
    if not math.isfinite(gain) or abs(gain) < 1e-12:
        raise RuntimeError("invalid source-derived model gain")
    return gain


def tile_starts(size):
    starts = list(range(0, max(1, size - TILE + 1), STEP))
    if size > TILE and starts[-1] != size - TILE:
        starts.append(size - TILE)
    return starts


def axis_weights(starts, index):
    weights = np.ones(TILE * 2, dtype=np.float32)
    if index:
        overlap = (starts[index - 1] + TILE - starts[index]) * 2
        weights[:overlap] = np.linspace(0, 1, overlap, endpoint=False)
    if index + 1 < len(starts):
        overlap = (starts[index] + TILE - starts[index + 1]) * 2
        weights[-overlap:] = np.linspace(1, 0, overlap, endpoint=False)
    return weights


def infer(source, session):
    height, width = source.shape[1:]
    output = np.zeros((3, height * 2, width * 2), dtype=np.float32)
    total = np.zeros((height * 2, width * 2), dtype=np.float32)
    ys, xs = tile_starts(height), tile_starts(width)
    name = session.get_inputs()[0].name
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            patch = source[:, y:y + TILE, x:x + TILE]
            h, w = patch.shape[1:]
            padded = np.pad(patch, ((0, 0), (0, TILE - h), (0, TILE - w)), mode="reflect")
            results = session.run(None, {name: np.ascontiguousarray(padded[None])})
            if len(results) != 1 or results[0].shape != (1, 3, TILE * 2, TILE * 2):
                raise RuntimeError("expected one float32 [1,3,1024,1024] model output")
            rgb = results[0][0, :, :h * 2, :w * 2]
            rgb = rgb * recover_gain(patch, rgb)
            weight = axis_weights(ys, yi)[:h * 2, None] * axis_weights(xs, xi)[None, :w * 2]
            output[:, y * 2:(y + h) * 2, x * 2:(x + w) * 2] += rgb * weight
            total[y * 2:(y + h) * 2, x * 2:(x + w) * 2] += weight
    if (total <= 0).any():
        raise RuntimeError("tiling left uncovered pixels")
    return output / total[None]


def metrics(gt, candidate):
    import colour
    import torch
    from pytorch_msssim import ms_ssim

    if (gt.shape != candidate.shape or not np.isfinite(gt).all()
            or not np.isfinite(candidate).all()):
        raise RuntimeError("invalid metric inputs")
    mse = float(np.mean((gt.astype(np.float64) - candidate) ** 2))
    with torch.inference_mode():
        similarity = float(ms_ssim(torch.from_numpy(candidate[None]),
                                   torch.from_numpy(gt[None]), data_range=1, size_average=True))
    numerator = denominator = 0.0
    for a, b in zip(gt, candidate):
        def response(channel):
            blurred = cv2.GaussianBlur(channel, (15, 15), 1.5,
                                       borderType=cv2.BORDER_REFLECT_101)
            return cv2.Laplacian(blurred, cv2.CV_32F, ksize=3,
                                 borderType=cv2.BORDER_REFLECT_101)[7:-7, 7:-7].astype(np.float64)
        ra, rb = response(a), response(b)
        numerator += float(np.sum((ra - rb) ** 2))
        denominator += float(np.sum(ra ** 2))
    if denominator <= 1e-20:
        raise RuntimeError("reference has insufficient texture for HFEN")
    xyz_matrix = np.linalg.inv(XYZ_TO_SRGB)
    def xyz(rgb):
        hwc = np.moveaxis(rgb, 0, -1)
        small = cv2.resize(hwc, (hwc.shape[1] // 8, hwc.shape[0] // 8),
                           interpolation=cv2.INTER_AREA)
        return small @ xyz_matrix.T
    reference_xyz, candidate_xyz = xyz(gt), xyz(candidate)
    support = cv2.resize((gt < 0.999).all(axis=0).astype(np.float32),
                         (reference_xyz.shape[1], reference_xyz.shape[0]),
                         interpolation=cv2.INTER_AREA) >= 0.999
    support &= reference_xyz[..., 1] > 0
    if support.mean() < 0.1:
        raise RuntimeError("too few unclipped reference pixels for color metrics")
    white = float(np.percentile(reference_xyz[..., 1][support], 99))
    support &= reference_xyz[..., 1] / white >= 0.01
    if support.mean() < 0.1:
        raise RuntimeError("too few non-dark reference pixels for color metrics")
    d65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    labs = [colour.XYZ_to_Lab(np.clip(values / white, 0, None), illuminant=d65)
            for values in (reference_xyz[support], candidate_xyz[support])]
    delta = float(np.mean(colour.delta_E(*labs, method="CIE 2000")))
    values = {"PSNR": 10 * math.log10(1 / mse) if mse else float("inf"),
              "MS-SSIM": similarity, "HFEN": math.sqrt(numerator / denominator), "Mean ΔE00": delta}
    if (math.isnan(values["PSNR"]) or any(not math.isfinite(values[key])
            for key in ("MS-SSIM", "HFEN", "Mean ΔE00"))):
        raise RuntimeError("metrics produced invalid values")
    return values


def write_report(path, rows, settings):
    lines = ["# Okeinla and RawNIND: direct Python RGB benchmark", "",
             "Dataset: [AyedaOk/rawbench-dataset](https://huggingface.co/datasets/AyedaOk/rawbench-dataset).",
             "Photos: Ayeda Okambawa, CC BY-SA 4.0. No training-data independence claim.", "",
             "Protocol: source-derived per-tile signed least-squares gain; 512 packed-pixel tiles,",
             "64 packed-pixel overlap, feathered blending. Fixed sensor crop x=[12,7020), y=[8,4680).",
             "Bilinear reference demosaic; common reference WB and D65-normalized linear-sRGB matrix.",
             "Source/reference integer alignment and exposure are shared across models; clip afterwards.",
             "PSNR/MS-SSIM/HFEN use linear RGB. HFEN: Gaussian 1.5/15, Laplacian 3, 7-pixel border.",
             "ΔE00: 8x area reduction, D65 Lab, reference 99th-percentile white, exclude clipped/dark support.",
             "Means weight each image equally. See settings for exact reference selections.", "",
             "## Results", ""]
    count = len({row["source"] for row in rows})
    lines += [f"Evaluated {count} of 133 noisy images.",
              "**Subset run; not a full-dataset result.**" if count < 133 else "Full-dataset run.", ""]
    groups = [("Overall", rows)]
    groups += [(f"ISO {iso}", [row for row in rows if row["iso"] == iso])
               for iso in sorted({row["iso"] for row in rows})]
    for title, group in groups:
        lines += [f"### {title}", "", "| Model | Images | PSNR ↑ | MS-SSIM ↑ | HFEN ↓ | Mean ΔE00 ↓ |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name in ("Okeinla", "RawNIND"):
            selected = [row["metrics"] for row in group if row["model"] == name]
            averages = [np.mean([row[key] for row in selected])
                        for key in ("PSNR", "MS-SSIM", "HFEN", "Mean ΔE00")]
            lines.append(f"| {name} | {len(selected)} | " + " | ".join(f"{v:.6f}" for v in averages) + " |")
        lines.append("")
    lines += ["## Reproducibility", "", "```json", json.dumps(settings, indent=2), "```", ""]
    temporary = path.with_suffix(".md.partial")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--okeinla", type=Path, default=Path("output/rawdenoise-okeinla/model_bayer.onnx"))
    parser.add_argument("--rawnind", type=Path, default=Path("output/rawdenoise-nind/model_bayer.onnx"))
    parser.add_argument("--output", type=Path, default=Path("output/benchmark"))
    parser.add_argument("--limit", type=int, help="score only the first N images (report marks the subset)")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    models = {"Okeinla": args.okeinla, "RawNIND": args.rawnind}
    for path in models.values():
        if not path.is_file():
            parser.error(f"missing ONNX: {path}; see benchmark/benchmark.md in the Okeinla repository for model setup")
    import onnxruntime as ort

    files, pairs = dataset_manifest()
    if args.limit:
        pairs = pairs[:args.limit]
    needed = {name for pair in pairs for name in pair}
    print(f"{len(pairs)} image pairs; {sum(files[n][1] for n in needed) / 1e9:.2f} GB RAW download before cache", flush=True)
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "rawpy", "onnxruntime", "opencv-python-headless", "torch", "pytorch-msssim", "colour-science")}
    settings = {"protocol": PROTOCOL, "dataset_revision": REVISION,
                "models": {name: sha256(path) for name, path in models.items()},
                "script_sha256": sha256(Path(__file__)),
                "alignment_sha256": sha256(Path(__file__).with_name("benchmark_alignment.py")),
                "versions": versions, "python": sys.version, "provider": "CPUExecutionProvider",
                "pairs": pairs, "references": REFERENCES, "limit": args.limit}
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    sessions = {name: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                for name, path in models.items()}
    for name, session in sessions.items():
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if (len(inputs) != 1 or inputs[0].shape != [1, 4, 512, 512]
                or inputs[0].type != "tensor(float)" or len(outputs) != 1
                or outputs[0].shape != [1, 3, 1024, 1024] or outputs[0].type != "tensor(float)"):
            raise RuntimeError(f"{name} must use the static FP32 Bayer contract")
    rows = []
    with sqlite3.connect(args.output / "metrics.sqlite") as database:
        database.execute("CREATE TABLE IF NOT EXISTS measurements "
                         "(run TEXT, source TEXT, model TEXT, result TEXT, PRIMARY KEY(run, source, model))")
        for index, (source, reference) in enumerate(pairs, 1):
            cached = {name: database.execute(
                "SELECT result FROM measurements WHERE run=? AND source=? AND model=?",
                (fingerprint, source, name)).fetchone() for name in models}
            if all(cached.values()):
                rows.extend(json.loads(cached[name][0]) for name in models)
                print(f"[{index}/{len(pairs)}] cached {source}", flush=True)
                continue
            paths = []
            for name in (source, reference):
                url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{urllib.parse.quote(name)}"
                paths.append(download(url, args.output / "data" / name, files[name][0]))
            noisy, _ = decode(paths[0])
            gt, profile = decode(paths[1])
            reference_rgb = convert_color(demosaic(gt), profile)
            source_rgb = convert_color(demosaic(noisy), profile)
            alignment = estimate_rgb_alignment(reference_rgb, source_rgb)
            del gt, source_rgb
            for name, session in sessions.items():
                if cached[name]:
                    rows.append(json.loads(cached[name][0]))
                    continue
                print(f"[{index}/{len(pairs)}] {name}: {source}", flush=True)
                rgb = convert_color(infer(noisy, session), profile)
                aligned_gt, aligned = apply_rgb_alignment(reference_rgb, rgb, alignment)
                result = {"source": source, "reference": reference, "model": name,
                          "iso": int(re.search(r"_ISO(\d+)_", source)[1]),
                          "alignment": asdict(alignment), "metrics": metrics(aligned_gt, aligned)}
                database.execute("INSERT INTO measurements VALUES (?, ?, ?, ?)",
                                 (fingerprint, source, name, json.dumps(result)))
                database.commit()
                rows.append(result)
                del rgb, aligned_gt, aligned
    write_report(args.output / "report.md", rows, settings)
    print(f"Report: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
