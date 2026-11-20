# Okeinla vs RawNIND benchmark

From the Okeinla checkout, save the script path. Use a separate `darktable-ai`
checkout for its locked Python environment and model converters:

```sh
benchmark="$PWD/benchmark/benchmark.py"
git clone https://github.com/AyedaOk/darktable-ai.git darktable-ai
cd darktable-ai
git checkout c0fb3ed
git submodule update --init vendor/rawnind_jddc
uv sync --group rawnind
uv run dtai setup rawdenoise-okeinla
uv run dtai convert rawdenoise-okeinla
uv run dtai setup rawdenoise-nind
uv run dtai convert rawdenoise-nind
```

Skip setup and conversion if you have both FP32 Bayer ONNX files in `output/`.
Run the benchmark:

```sh
uv run --group rawnind python "$benchmark"
```

The full run downloads about 5.7 GB and scores 133 images on CPU. To check RAM
usage first, add `--limit 1 --output output/benchmark-smoke`.

## Results and reruns

- `output/benchmark/report.md`: overall and per-ISO means, protocol and provenance.
- `output/benchmark/metrics.sqlite`: resumable per-image measurements.
- `output/benchmark/data/`: downloaded RAWs.

Rerun the same command after interruption to reuse downloads and matching
measurements. Changes to code, models, dataset selections or package versions
invalidate cached measurements. The report appears when the selected run finishes.
Do not run two copies against the same output directory.

Use `--okeinla PATH --rawnind PATH` to select other prepared ONNX files.

## Scoring

The benchmark compares both models' RGB predictions against base-ISO references
using PSNR, MS-SSIM, HFEN and ΔE00. Processing uses per-tile gain correction and
overlap blending, a common color transform, and shared alignment and exposure.
The reference uses bilinear demosaic. See `report.md` for the full settings.

The dataset covers one camera, the Sony A7C II. This benchmark does not establish
independence from either model's training data.

## Dataset

Downloads use [AyedaOk/rawbench-dataset](https://huggingface.co/datasets/AyedaOk/rawbench-dataset)
at revision `978c222c700b2dda9b4aa96af5fc54dad1fa6e87`, with SHA256 verification.
Reference selections for scenes 0001 through 0015:

```text
ISO100_2, ISO100_3, ISO100_2, ISO100_2, ISO100_1,
ISO100_3, ISO100_2, ISO100_3, ISO100_3, ISO100_1,
ISO100_1, ISO100_1, ISO100_1, ISO100_3, ISO100_3
```

Photos: Ayeda Okambawa, CC BY-SA 4.0; see the dataset's license.
Benchmark code: Ayeda Okambawa, under this repository's GPLv3 license.
