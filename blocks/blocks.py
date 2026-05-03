# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.nn import MODELS


class Permute(nn.Module):

    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, x):
        return x.transpose(-1, -2)


@MODELS.register()
class ConvPyramid(nn.Module):

    def __init__(self, dims, strides):
        super(ConvPyramid, self).__init__()

        self.blocks = nn.ModuleList()
        for s in strides:
            p = int(math.log2(s))
            if p == 0:
                layers = nn.ReLU(inplace=False)
            else:
                layers = nn.Sequential()
                conv_cls = nn.Conv1d if p > 0 else nn.ConvTranspose1d
                for _ in range(abs(p)):
                    layers.extend([
                        Permute(),
                        conv_cls(dims, dims, 2, stride=2),
                        Permute(),
                        nn.LayerNorm(dims),
                        nn.ReLU(inplace=True)
                    ])
            self.blocks.append(layers)

        self.strides = strides

    def forward(self, x, mask, return_mask=False):
        pymid, pymid_msk = [], []

        for s, blk in zip(self.strides, self.blocks):
            if x.size(1) < s:
                continue

            pymid.append(blk(x))

            if return_mask:
                if s > 1:
                    msk = F.max_pool1d(mask.float(), s, stride=s).long()
                elif s < 1:
                    msk = mask.repeat_interleave(int(1 / s), dim=1)
                else:
                    msk = mask
                pymid_msk.append(msk)

        return pymid, pymid_msk


@MODELS.register()
class AdaPooling(nn.Module):

    def __init__(self, dims):
        super(AdaPooling, self).__init__()
        self.att = nn.Linear(dims, 1, bias=False)

    def forward(self, x, mask):
        a = self.att(x) + torch.where(mask.unsqueeze(2) == 1, .0, float('-inf'))
        a = a.softmax(dim=1)
        x = torch.matmul(x.transpose(1, 2), a)
        x = x.squeeze(2).unsqueeze(1)
        return x



@MODELS.register()
class ConvHead(nn.Module):

    def __init__(self, dims, out_dims, kernal_size=3):
        super(ConvHead, self).__init__()

        # yapf:disable
        self.module = nn.Sequential(
            Permute(),
            nn.Conv1d(dims, dims, kernal_size, padding=kernal_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(dims, out_dims, kernal_size, padding=kernal_size // 2),
            Permute())
        # yapf:enable

    def forward(self, x):
        return self.module(x)


@MODELS.register()
class ConvPyramidFPN(ConvPyramid):
    """ConvPyramid with Feature Pyramid Network top-down pathway.

    Bottom-up (inherited): produces features at strides (1, 2, 4, 8).
    Top-down (added):   upsamples coarse features and fuses with fine
                         features via lateral connections, enabling
                         short-moment levels to receive long-range context.
    """

    def __init__(self, dims, strides):
        super().__init__(dims, strides)
        num_levels = len(strides)

        # Lateral 1x1 conv to align channels before element-wise fusion
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(Permute(), nn.Conv1d(dims, dims, 1),
                          Permute(), nn.LayerNorm(dims))
            for _ in range(num_levels)
        ])

        # Output 3x3 conv after each merge to reduce aliasing
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(Permute(), nn.Conv1d(dims, dims, 3, padding=1),
                          Permute(), nn.LayerNorm(dims), nn.ReLU(inplace=True))
            for _ in range(num_levels)
        ])

    def forward(self, x, mask, return_mask=False):
        # ---- bottom-up (inherited) ----
        pymid, pymid_msk = super().forward(x, mask, return_mask=True)
        num_levels = len(pymid)

        # ---- top-down (from coarsest to finest) ----
        fpn_feats = [None] * num_levels

        # start from coarsest: lateral → output conv
        p = self.lateral_convs[-1](pymid[-1])
        fpn_feats[-1] = self.fpn_convs[-1](p)

        for i in range(num_levels - 2, -1, -1):
            lat = self.lateral_convs[i](pymid[i])
            prev = fpn_feats[i + 1]

            # upsample coarser feature to current temporal resolution
            prev_up = F.interpolate(
                prev.transpose(1, 2),  # (B, C, T_{i+1})
                size=pymid[i].size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)          # (B, T_i, C)

            fpn_feats[i] = self.fpn_convs[i](lat + prev_up)

        if not return_mask:
            return fpn_feats, []

        # align masks to FPN output sizes
        fpn_msk = []
        for i in range(num_levels):
            m = pymid_msk[i] if i < len(pymid_msk) else pymid_msk[-1]
            if m.size(1) != fpn_feats[i].size(1):
                m = F.interpolate(
                    m.float().unsqueeze(1),
                    size=fpn_feats[i].size(1),
                    mode="nearest",
                ).squeeze(1).long()
            fpn_msk.append(m)

        return fpn_feats, fpn_msk


@MODELS.register()
class ConvPyramidChainedFPN(nn.Module):
    """True hierarchical feature pyramid with chained bottom-up + FPN top-down.

    Bottom-up (chained):  x → C₁ → ds → C₂ → ds → C₃ → ds → C₄
    Top-down (FPN):       C₄ → ×2 → +C₃ → ... → P₁, P₂, P₃, P₄

    Unlike ConvPyramidFPN (which applies independent blocks to x),
    each level builds on the previous, creating genuine semantic
    hierarchy — coarser levels encode progressively more abstract
    features, making the top-down pathway meaningful.
    """

    def __init__(self, dims, strides):
        super().__init__()
        self.strides = strides
        num_levels = len(strides)

        self.blocks = nn.ModuleList()
        for s in strides:
            p = int(math.log2(s))
            if p == 0:
                layers = nn.ReLU(inplace=False)
            else:
                layers = nn.Sequential()
                for _ in range(abs(p)):
                    layers.extend([
                        Permute(),
                        nn.Conv1d(dims, dims, 3, stride=2, padding=1),
                        Permute(),
                        nn.LayerNorm(dims),
                        nn.ReLU(inplace=True),
                    ])
            self.blocks.append(layers)

        # FPN lateral: 1x1 conv
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(Permute(), nn.Conv1d(dims, dims, 1),
                          Permute(), nn.LayerNorm(dims))
            for _ in range(num_levels)
        ])

        # FPN output: 3x3 conv after merge
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(Permute(), nn.Conv1d(dims, dims, 3, padding=1),
                          Permute(), nn.LayerNorm(dims), nn.ReLU(inplace=True))
            for _ in range(num_levels)
        ])

    def forward(self, x, mask, return_mask=False):
        # ---- chained bottom-up ----
        pymid, pymid_msk = [], []
        cur = x
        cur_mask = mask

        for s, blk in zip(self.strides, self.blocks):
            if cur.size(1) < s:
                continue
            cur = blk(cur)
            pymid.append(cur)

            if return_mask:
                if s > 1:
                    cur_mask = F.max_pool1d(cur_mask.float(), 2, stride=2).long()
                pymid_msk.append(cur_mask)

        num_levels = len(pymid)

        # ---- FPN top-down ----
        fpn_feats = [None] * num_levels
        p = self.lateral_convs[-1](pymid[-1])
        fpn_feats[-1] = self.fpn_convs[-1](p)

        for i in range(num_levels - 2, -1, -1):
            lat = self.lateral_convs[i](pymid[i])
            prev_up = F.interpolate(
                fpn_feats[i + 1].transpose(1, 2),
                size=pymid[i].size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            fpn_feats[i] = self.fpn_convs[i](lat + prev_up)

        if not return_mask:
            return fpn_feats, []

        return fpn_feats, pymid_msk
