# saev - Sparse Auto-Encoders for Vision

![PyPI Downloads](https://static.pepy.tech/badge/saev)
![MIT License](https://img.shields.io/badge/License-MIT-efefef)
![GitHub Repo stars](https://img.shields.io/github/stars/OSU-NLP-group/saev?style=flat&label=GitHub%20%E2%AD%90)


Sparse autoencoders (SAEs) for vision transformers (ViTs), implemented in PyTorch.

This is the codebase used for our preprint "Sparse Autoencoders for Scientifically Rigorous Interpretation of Vision Models"

* [arXiv preprint](https://arxiv.org/abs/2502.06755)
* [Huggingface Models](https://huggingface.co/collections/osunlp/sae-v-67ab8c4fdf179d117db28195)
* [API Docs](https://osu-nlp-group.github.io/saev/api/saev)
* [Demos](https://osu-nlp-group.github.io/saev/#demos)

## About

saev is a package for training sparse autoencoders (SAEs) on vision transformers (ViTs) in PyTorch.
It also includes an interactive webapp for looking through a trained SAE's features.

Originally forked from [HugoFry](https://github.com/HugoFry/mats_sae_training_for_ViTs) who forked it from [Joseph Bloom](https://github.com/jbloomAus/SAELens).

Read [logbook.md](docs/research/logbook.md) for a detailed log of my thought process.

See [related-work.md](saev/related-work.md) for a list of works training SAEs on vision models.
Please open an issue or a PR if there is missing work.

## Installation

Installation is supported with [uv](https://docs.astral.sh/uv/).
saev will likely work with pure pip, conda, etc. but I will not formally support it.

Clone this repository, then from the root directory:

```bash
uv run python -m saev --help
```

This will create a virtual environment and display the CLI help.

## Using `saev`

See the [docs](https://osu-nlp-group.github.io/saev/api/saev) for an overview.

You can ask questions about this repo using the `llms.txt` file.

Example (macOS):

`curl https://osu-nlp-group.github.io/saev/api/llms.txt | pbcopy`, then paste into [Claude](https://claude.ai) or any LLM interface of your choice.
