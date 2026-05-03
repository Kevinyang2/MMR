import logging
import os
import random
from collections import defaultdict

import nncore
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from FlashMMR.config import BaseOptions
from FlashMMR.inference import eval_epoch
from FlashMMR.model import build_model
from FlashMMR.start_end_dataset import (
    StartEndDataset,
    prepare_batch_inputs,
    start_end_collate,
)
from utils.basic_utils import AverageMeter, save_json


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataset(opt, data_path, is_train):
    if data_path is None:
        return None
    return StartEndDataset(
        dset_name=opt.dset_name,
        data_path=data_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type=opt.q_feat_type,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio if is_train else 1.0,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=True,
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=opt.txt_drop_ratio if is_train else 0,
        dset_domain=opt.dset_domain,
    )


def setup_model_optimizer(opt):
    model = build_model(opt)
    if opt.device.type == "cuda":
        model.to(opt.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_drop, gamma=0.1)

    start_epoch = 0
    if opt.resume is not None:
        logger.info("Load checkpoint from %s", opt.resume)
        checkpoint = torch.load(opt.resume, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

        if opt.resume_all:
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint.get("epoch", -1)) + 1

    if opt.start_epoch is not None:
        start_epoch = opt.start_epoch

    return model, optimizer, scheduler, start_epoch


def weighted_loss(loss_dict, opt):
    weights = {
        "loss_cls": opt.lw_cls,
        "loss_reg": opt.lw_reg,
        "loss_sal": opt.lw_sal,
        "loss_pv": opt.lw_pv,
        "loss_pv_repr": opt.lw_pv1,
        "loss_pv_adj": opt.lw_pv_adj,
    }
    return sum(loss_dict[k] * weights.get(k, 1.0) for k in loss_dict)


def train_one_epoch(model, train_loader, optimizer, opt, epoch, tb_writer=None):
    model.train()
    meters = defaultdict(AverageMeter)

    for batch in tqdm(train_loader, desc=f"train epoch {epoch + 1}"):
        model_inputs, targets = prepare_batch_inputs(
            batch[1], opt.device, non_blocking=opt.pin_memory
        )
        outputs = model(**model_inputs, targets=targets)
        loss_dict = {k: v for k, v in outputs.items() if k.startswith("loss_")}
        loss = weighted_loss(loss_dict, opt)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if opt.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        meters["loss_overall"].update(float(loss.detach().cpu()))
        for k, v in loss_dict.items():
            meters[k].update(float(v.detach().cpu()))

    logger.info(
        "epoch %d train %s",
        epoch + 1,
        " ".join(f"{k}={v.avg:.4f}" for k, v in meters.items()),
    )
    if tb_writer is not None:
        for k, v in meters.items():
            tb_writer.add_scalar(f"Train/{k}", v.avg, epoch + 1)

    return meters


def save_checkpoint(model, optimizer, scheduler, opt, epoch, is_best=False):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "opt": vars(opt),
    }
    torch.save(checkpoint, opt.ckpt_filepath)
    if is_best:
        torch.save(checkpoint, os.path.join(opt.results_dir, "model_best.ckpt"))


def metric_for_selection(metrics):
    if metrics is None:
        return None
    brief = metrics.get("brief", metrics)
    return brief.get("MR-full-mAP", None)


def main():
    opt = BaseOptions().parse()
    opt.cfg = nncore.Config.from_file(opt.config)
    set_seed(opt.seed)

    train_dataset = make_dataset(opt, opt.train_path, is_train=True)
    if train_dataset is None:
        raise ValueError("--train_path is required for training")
    eval_dataset = make_dataset(opt, opt.eval_path, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        drop_last=opt.drop_last,
        pin_memory=opt.pin_memory,
    )

    model, optimizer, scheduler, start_epoch = setup_model_optimizer(opt)
    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    best_score = float("-inf")
    bad_epochs = 0

    for epoch in range(start_epoch, opt.n_epoch):
        train_one_epoch(model, train_loader, optimizer, opt, epoch, tb_writer)
        scheduler.step()
        save_checkpoint(model, optimizer, scheduler, opt, epoch, is_best=False)

        if eval_dataset is not None and (epoch + 1) % opt.eval_epoch == 0:
            metrics, metrics_nms, eval_loss_meters, latest_file_paths = eval_epoch(
                model,
                eval_dataset,
                opt,
                save_submission_filename=f"hl_val_epoch_{epoch + 1}_submission.jsonl",
                epoch_i=epoch,
                tb_writer=tb_writer,
            )
            score = metric_for_selection(metrics_nms) or metric_for_selection(metrics)
            if score is not None and score > best_score:
                best_score = score
                bad_epochs = 0
                save_checkpoint(model, optimizer, scheduler, opt, epoch, is_best=True)
                save_json({"best_epoch": epoch + 1, "best_score": best_score}, os.path.join(opt.results_dir, "best.json"))
            else:
                bad_epochs += 1

            if opt.max_es_cnt > 0 and bad_epochs >= opt.max_es_cnt:
                logger.info("Early stop at epoch %d", epoch + 1)
                break

    tb_writer.close()


if __name__ == "__main__":
    main()
