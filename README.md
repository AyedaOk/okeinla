# Okeinla

Okeinla is a Bayer RAW denoising and demosaicing model built on
[RawNIND](https://github.com/trougnouf/rawnind_jddc) and NAFNet.

- [Model weights and application packages](https://huggingface.co/AyedaOk/okeinla)
- [Okeinla dataset](https://huggingface.co/datasets/AyedaOk/okeinla-dataset)

## Setup

Use Python 3.13 or newer and [uv](https://docs.astral.sh/uv/):

```sh
uv sync
cd src/rawnind
export OPENCV_IO_ENABLE_OPENEXR=1
```

Run training and tools from `src/rawnind`.

## Credits

- **RawNIND:** Benoit Brummer and Christophe De Vleeschouwer,
  *Learning Joint Denoising, Demosaicing, and Compression from the Raw Natural
  Image Noise Dataset* (2025), [paper](https://arxiv.org/abs/2501.08924).
- **RawForge / Restorer:** Okeinla uses a loss recipe derived from its
  [training code](https://github.com/rymuelle/Restorer).
- **NAFNet:** Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, and Jian Sun,
  *Simple Baselines for Image Restoration* (ECCV 2022),
  [code](https://github.com/megvii-research/NAFNet).

## License

Code and model weights: GPLv3. Okeinla and RawNIND datasets: CC BY-SA 4.0.
See [LICENSE](LICENSE) and the [NAFNet MIT notice](licenses/NAFNet-MIT.txt).
