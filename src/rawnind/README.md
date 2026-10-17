# Okeinla source

Run training and tools from this directory. See the
[project README](../../README.md).

`train_denoiser_bayer2prgb.py` is the Okeinla training entry point.
`models/nafnet.py` defines its network. Shared RAW processing lives in `libs/`;
`../common/` contains image utilities and losses.
