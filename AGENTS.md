# AGENTS.md

## Source of Truth
- Trust executable files over prose: `FlashMMR/config.py` defines CLI/defaults, `FlashMMR/train.py` defines checkpoint/eval flow, and `data/MR.py` defines the nncore model config loaded by `python FlashMMR/train.py data/MR.py ...`.
- No repo-local lint, typecheck, test, CI, or formatter config exists; verify changes with focused imports/short runs rather than inventing generic commands.
- Treat `FlashMMR/` and `CGSTVG/` as separate training stacks; do not mix their config, checkpoint, or dataset conventions.

## Environment and Paths
- The documented QV-M3/FlashMMR environment is `conda` env `flashmmr` with Python 3.12; install with `pip install -r requirements.txt` (`torch==2.2.2`, `torchtext==0.17.2`, `nncore==0.4.2`).
- `CGSTVG/` has its own `CGSTVG/requirements.txt` (`torchtext==0.15.2`, `transformers==4.5.1`, `yacs`, OpenCV/ffmpeg deps); do not assume the root requirements satisfy that stack.
- Scripts assume the repo root is importable. On Windows use `$env:PYTHONPATH = "D:\QV-M3"`; shell scripts use `PYTHONPATH=$PYTHONPATH:.`.
- `features/` is gitignored and expected to contain QV-M2 feature dirs such as `slowfast_features`, `clip_features`, and `clip_text_features_new`.

## FlashMMR Workflows
- Baseline training uses `conda run -n flashmmr --no-capture-output python FlashMMR/train.py data/MR.py ...`; `train.md` contains the canonical long argument set.
- `--exp_id` is required for training. Results are written under `results/<dset>-<ctx>-<exp_id>-<timestamp>/` with `opt.json`, `model.ckpt`, `model_best.ckpt`, `best.json`, `code.zip`, TensorBoard logs, and per-eval submission/metrics files.
- Standalone metric recomputation is `python standalone_eval/eval.py --submission_path <jsonl> --gt_path data/QV-M2/test.jsonl --save_path <json>`.

## FlashMMR Architecture Notes
- `FlashMMR/train.py` builds `StartEndDataset`, `FlashMMR.model.build_model`, AdamW, StepLR, and calls `FlashMMR.inference.eval_epoch` every `--eval_epoch` epochs.
- Model variants are selected through `data/MR.py` nncore config: `pyramid_cfg=dict(type="ConvPyramid")`, heads/losses in `blocks/*`, and runtime CLI knobs in `FlashMMR/config.py`.
- `blocks/blocks.py` registers the baseline `ConvPyramid`; switch model wiring through `data/MR.py` rather than editing generated copies in `results/`.
- `FlashMMR/inference.py` writes both raw submissions and NMS submissions by default (`--nms_thd` defaults to `0.7`; use `-1` to disable NMS).

## CGSTVG Notes
- `CGSTVG/` is a separate YACS/distributed Torch project with entrypoints `CGSTVG/scripts/train_net.py` and `CGSTVG/scripts/test_net.py`; configs live in `CGSTVG/experiments/*.yaml` and `CGSTVG/config/defaults.py`.
- CGSTVG datasets are selected in `CGSTVG/datasets/build.py` (`VidSTG` or `HC-STVG`), with `SOLVER.BATCH_SIZE == 1` asserted per GPU.
- CGSTVG checkpoints are `.pth` files plus `last_checkpoint` under `OUTPUT_DIR` via `CGSTVG/utils/checkpoint.py`, not FlashMMR `model.ckpt` files.

## Artifact Guardrails
- Do not edit or review generated files under `results/`, `eval_results/`, `debug_results/`, `features/`, `__pycache__/`, or CGSTVG `data/*/checkpoints/` unless explicitly asked; they are outputs or local data.
- FlashMMR training snapshots copy `FlashMMR/model.py` and `FlashMMR/transformer.py` into each result directory. Treat those copies as provenance, not source.
