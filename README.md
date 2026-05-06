# QV-M3: Multi-Moment Retrieval with Fine-Grained Temporal Modeling

基于 QV-M2 (NeurIPS 2025) FlashMMR 最优配置（G-mAP 35.06），聚焦短时刻检测改进。

## Baseline

| Metric | QV-M2 FlashMMR |
|---|---|
| G-mAP | 35.06 |
| 1-tgt mAP | 51.89 |
| 2-tgt mAP | 45.48 |
| 3+tgt mAP | 22.59 |
| mIoU@1 | 56.19 |
| mR@1 | 49.42 |
| Short (0-10s) | 15.90 |

## 快速开始

```bash
conda create -n flashmmr python=3.12 -y && conda activate flashmmr
pip install -r requirements.txt
```

## 训练

```bash
python FlashMMR/train.py data/MR.py \
    --exp_id flashMMR_baseline \
    --use_neg --dset_name hl --ctx_mode video_tef \
    --train_path data/QV-M2/train.jsonl --eval_path data/QV-M2/test.jsonl \
    --v_feat_dirs features/slowfast_features features/clip_features --v_feat_dim 2816 \
    --t_feat_dir features/clip_text_features_new/ --t_feat_dim 512 \
    --max_v_l 75 --max_q_l 40 --max_windows 5 \
    --bsz 64 --n_epoch 150 --eval_bsz 1 --eval_epoch 3 \
    --use_SRM --use_pv_repr \
    --kernel_size 5 --num_conv_layers 1 --num_mlp_layers 5 \
    --t2v_layers 6 --num_dummies 40 \
    --lw_reg 1.0 --lw_cls 5.0 --lw_saliency 0.8 \
    --lw_pv 9.0 --lw_pv1 0.7
```

## 评估

```bash
python standalone_eval/eval.py \
    --submission_path results/<exp>/hl_val_epoch_<N>_submission_nms_thd_0.7.jsonl \
    --gt_path data/QV-M2/test.jsonl \
    --save_path results/<exp>/metrics.json
```