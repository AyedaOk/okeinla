"""Export a trained Bayer denoiser as ONNX and a darktable .dtmodel archive."""

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))

from rawnind.models import nafnet, raw_denoiser


DEFAULT_METRIC = "val_rawforge_loss.None"


def load_experiment_args(model_dir: Path) -> dict:
    args_path = model_dir / "args.yaml"
    if not args_path.is_file():
        raise FileNotFoundError(f"experiment args not found: {args_path}")
    with args_path.open() as file:
        return yaml.safe_load(file)


def resolve_checkpoint(model_dir: Path, checkpoint: Path | None, metric: str) -> Path:
    if checkpoint is not None:
        checkpoint = checkpoint.resolve()
    else:
        results_path = model_dir / "trainres.yaml"
        if not results_path.is_file():
            raise FileNotFoundError(f"training results not found: {results_path}")
        with results_path.open() as file:
            results = yaml.safe_load(file)
        try:
            step = results["best_step"][metric]
        except KeyError as error:
            raise KeyError(f"metric {metric!r} not found in {results_path}") from error
        checkpoint = model_dir / "saved_models" / f"iter_{step}.pt"

    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    return checkpoint


def validate_experiment_args(args: dict) -> None:
    if args.get("in_channels") != 4:
        raise ValueError("darktable Bayer export requires in_channels: 4")
    if args.get("preupsample", False):
        raise ValueError("darktable Bayer export does not support preupsample models")
    if args.get("iso_conditioning", False):
        raise ValueError(
            "darktable 5.6 does not provide the ISO input required by this model"
        )
    if args.get("match_gain") not in ("output", "target"):
        raise ValueError("darktable manifest requires training with match_gain: output or target")
    if args.get("transfer_function") not in (None, "None"):
        raise ValueError("darktable Bayer export requires a linear input transfer function")


def build_model(args: dict) -> torch.nn.Module:
    validate_experiment_args(args)
    model_kwargs = {
        "in_channels": args["in_channels"],
        "funit": args["funit"],
        "preupsample": args.get("preupsample", False),
    }
    architectures = {
        "unet": raw_denoiser.UtNet2,
        "nafnet": nafnet.NAFNet,
    }
    try:
        model_class = architectures[args["arch"]]
    except KeyError as error:
        raise ValueError(
            f"unsupported darktable export architecture: {args.get('arch')!r}"
        ) from error
    return model_class(**model_kwargs).eval()


def load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model_state_dict", "params", "params_ema"):
        if isinstance(state_dict, dict) and key in state_dict:
            state_dict = state_dict[key]
            break
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint does not contain a state dictionary: {checkpoint}")
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)


def export_onnx(model: torch.nn.Module, output: Path, tile_size: int, opset: int) -> None:
    if tile_size <= 0 or tile_size % 16:
        raise ValueError("tile size must be a positive multiple of 16")
    sample = torch.zeros(1, 4, tile_size, tile_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        sample,
        output,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )


def validate_onnx_parity(
    model: torch.nn.Module,
    onnx_path: Path,
    tile_size: int,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> None:
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    model_input = session.get_inputs()
    model_output = session.get_outputs()
    expected_input = [1, 4, tile_size, tile_size]
    expected_output = [1, 3, tile_size * 2, tile_size * 2]
    if len(model_input) != 1 or model_input[0].shape != expected_input:
        raise ValueError(
            f"unexpected ONNX input metadata: {[(item.name, item.shape) for item in model_input]}"
        )
    if len(model_output) != 1 or model_output[0].shape != expected_output:
        raise ValueError(
            f"unexpected ONNX output metadata: {[(item.name, item.shape) for item in model_output]}"
        )

    generator = torch.Generator().manual_seed(2026)
    sample = torch.rand(1, 4, tile_size, tile_size, generator=generator)
    with torch.inference_mode():
        torch_output = model(sample).numpy()
    [onnx_output] = session.run(None, {model_input[0].name: sample.numpy()})
    if not np.isfinite(onnx_output).all():
        raise ValueError("ONNX output contains non-finite values")
    np.testing.assert_allclose(onnx_output, torch_output, atol=atol, rtol=rtol)


def checkpoint_sha256(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    with checkpoint.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_model_id(value: str) -> str:
    model_id = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not model_id:
        raise ValueError("model id must contain at least one letter or number")
    return model_id


def make_config(
    args: dict,
    model_id: str,
    name: str,
    description: str,
    version: str,
    tile_size: int,
    checkpoint: Path,
) -> dict:
    darktable_arch = "utnet2" if args["arch"] == "unet" else args["arch"]
    output_scale = "match_gain" if args.get("match_gain") == "output" else "none"
    return {
        "id": model_id,
        "name": name,
        "description": description,
        "task": "rawdenoise",
        "arch": darktable_arch,
        "backend": "onnx",
        "version": version,
        "tiling": True,
        "model_card": {
            "long_description": description,
            "scope": "Bayer raw denoising and demosaicing",
            "source": "https://github.com/trougnouf/rawnind_jddc",
            "license": "GPL-3.0",
            "training_code": "Okeinla (derived from RawNIND)",
            "author": "Ayeda O",
            "experiment": args.get("expname", "unknown"),
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": checkpoint_sha256(checkpoint),
        },
        "attributes": {
            "input_sizes": [tile_size],
            "model_bayer": {
                "input_kind": "bayer_v1",
                "bayer_orientation": "force_rggb",
                "edge_pad": "mirror_cropped",
                "wb_norm": "none",
                "output_scale": output_scale,
            },
        },
    }


def package_dtmodel(package_dir: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(package_dir.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(package_dir.parent))


def export_darktable_model(
    model_dir: Path,
    output_dir: Path,
    checkpoint: Path | None = None,
    metric: str = DEFAULT_METRIC,
    model_id: str | None = None,
    name: str | None = None,
    description: str | None = None,
    version: str = "1.0",
    tile_size: int = 512,
    opset: int = 20,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    model_dir = model_dir.resolve()
    args = load_experiment_args(model_dir)
    checkpoint = resolve_checkpoint(model_dir, checkpoint, metric)
    model = build_model(args)
    load_checkpoint(model, checkpoint)

    model_id = make_model_id(model_id or args.get("expname", model_dir.name))
    name = name or model_id
    description = description or (
        f"{args['arch']} Bayer denoiser trained by RawNIND experiment "
        f"{args.get('expname', model_dir.name)}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    package_dir = output_dir / model_id
    archive_path = output_dir / f"{model_id}.dtmodel"
    if not overwrite and (package_dir.exists() or archive_path.exists()):
        raise FileExistsError(
            f"output exists; pass --overwrite to replace {package_dir} and {archive_path}"
        )

    with tempfile.TemporaryDirectory(prefix=f".{model_id}-", dir=output_dir) as temp:
        temp_package_dir = Path(temp) / model_id
        temp_package_dir.mkdir()
        onnx_path = temp_package_dir / "model_bayer.onnx"
        export_onnx(model, onnx_path, tile_size, opset)
        validate_onnx_parity(model, onnx_path, tile_size)
        config = make_config(
            args,
            model_id,
            name,
            description,
            version,
            tile_size,
            checkpoint,
        )
        (temp_package_dir / "config.json").write_text(
            json.dumps(config, indent=4, ensure_ascii=False) + "\n"
        )
        temp_archive = Path(temp) / f"{model_id}.dtmodel"
        package_dtmodel(temp_package_dir, temp_archive)

        if overwrite:
            if package_dir.exists():
                shutil.rmtree(package_dir)
            archive_path.unlink(missing_ok=True)
        shutil.move(temp_package_dir, package_dir)
        shutil.move(temp_archive, archive_path)

    return package_dir, archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--id", dest="model_id")
    parser.add_argument("--name")
    parser.add_argument("--description")
    parser.add_argument("--version", default="1.0")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    package_dir, archive_path = export_darktable_model(
        model_dir=arguments.model_dir,
        output_dir=arguments.output_dir,
        checkpoint=arguments.checkpoint,
        metric=arguments.metric,
        model_id=arguments.model_id,
        name=arguments.name,
        description=arguments.description,
        version=arguments.version,
        tile_size=arguments.tile_size,
        opset=arguments.opset,
        overwrite=arguments.overwrite,
    )
    print(f"darktable model directory: {package_dir}")
    print(f"darktable package: {archive_path}")


if __name__ == "__main__":
    main()
