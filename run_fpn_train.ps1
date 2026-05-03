# Run from D:\QV-M3 root directory
Set-Location D:\QV-M3
$env:PYTHONPATH = "D:\QV-M3"
conda activate flashmmr
python FlashMMR/train.py data/MR.py --exp_id flashMMR_fpn --use_neg --dset_name hl --ctx_mode video_tef --train_path D:\QV-M3\data\QV-M2\train.jsonl --eval_path D:\QV-M3\data\QV-M2\test.jsonl --v_feat_dirs D:\QV-M3\features\slowfast_features D:\QV-M3\features\clip_features --v_feat_dim 2816 --t_feat_dir D:\QV-M3\features\clip_text_features_new\ --t_feat_dim 512 --max_v_l 75 --max_q_l 40 --max_windows 5 --bsz 64 --n_epoch 150 --eval_bsz 1 --eval_epoch 3 --use_SRM --use_pv_repr --kernel_size 5 --num_conv_layers 1 --num_mlp_layers 5 --t2v_layers 6 --num_dummies 40 --lw_reg 1.0 --lw_cls 5.0 --lw_saliency 0.8 --lw_pv 9.0 --lw_pv1 0.7
