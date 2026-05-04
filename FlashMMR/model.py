# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
FlashMMR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from FlashMMR.transformer import build_transformer, TransformerEncoderLayer, TransformerEncoder
from FlashMMR.position_encoding import build_position_encoding, PositionEmbeddingSine
from FlashMMR.span_utils import span_cxw_to_xx
from nncore.nn import build_model as build_adapter
from nncore.nn import build_loss
from blocks.generator import PointGenerator



def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def find_nth(vid, underline, n):
    max_len = len(vid)
    start = vid.find(underline)
    while start >= 0 and n > 1:
        start = vid.find(underline, start+len(underline))
        n -= 1
    if start == -1:
        start = max_len
    return start

def element_wise_list_equal(listA, listB):
    res = []
    for a, b in zip(listA, listB):
        if a==b:
            res.append(True)
        else:
            res.append(False)
    return res

class ConfidenceScorer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_conv_layers=1, num_mlp_layers=3):
        super(ConfidenceScorer, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        for i in range(num_conv_layers):
            if i == 0:
                self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=(0, kernel_size[1] // 2)))
            else:
                self.convs.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=(0, kernel_size[1] // 2)))
            self.activations.append(nn.ReLU(inplace=True))
        
        self.fc = MLP(out_channels, out_channels // 2, 1, num_layers=num_mlp_layers)
    
    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        
        for conv, activation in zip(self.convs, self.activations):
            x = conv(x)
            x = activation(x)
        
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.fc(x)
        
        return x

class FlashMMR(nn.Module):
    """ FlashMMR. """

    def __init__(self, transformer, position_embed, txt_position_embed, n_input_proj, input_dropout, txt_dim, vid_dim, aud_dim=0, use_txt_pos=False,
                strides=(1, 2, 4, 8),
                buffer_size=2048,
                max_num_moment=50,
                merge_cls_sal=True,
                pyramid_cfg=None,
                coord_head_cfg=None,
                loss_cfg=None,
                args=None):
        """ Initializes the model."""
        super().__init__()
        self.args=args
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.PositionEmbeddingSine = PositionEmbeddingSine(hidden_dim, normalize=True)
        
        # input projection
        self.n_input_proj = n_input_proj
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        # set up dummy token
        self.token_type_embeddings = nn.Embedding(2, hidden_dim)
        self.token_type_embeddings.apply(init_weights)
        self.use_txt_pos = use_txt_pos
        self.dummy_rep_token = torch.nn.Parameter(torch.randn(args.num_dummies, hidden_dim))
        self.dummy_rep_pos = torch.nn.Parameter(torch.randn(args.num_dummies, hidden_dim))
        normalize_before = False
        input_txt_sa_proj = TransformerEncoderLayer(hidden_dim, 8, self.args.dim_feedforward, 0.1, "prelu", normalize_before)
        txtproj_encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.txtproj_encoder = TransformerEncoder(input_txt_sa_proj, args.dummy_layers, txtproj_encoder_norm)

        # build muti-scale pyramid
        self.pyramid = build_adapter(pyramid_cfg, hidden_dim, strides)
        self.conf_head = ConfidenceScorer(in_channels=256, out_channels=256, kernel_size=(1, args.kernel_size), num_conv_layers=args.num_conv_layers, num_mlp_layers = args.num_mlp_layers)
        self.class_head = ConfidenceScorer(in_channels=256, out_channels=256, kernel_size=(1, args.kernel_size), num_conv_layers=args.num_conv_layers, num_mlp_layers = args.num_mlp_layers)
        self.coef = nn.Parameter(torch.ones(len(strides)))
        self.coord_head = build_adapter(coord_head_cfg, hidden_dim, 2)
        self.criterion = build_loss(loss_cfg) if loss_cfg is not None else None
        self.generator = PointGenerator(strides, buffer_size)
        self.max_num_moment = max_num_moment
        self.merge_cls_sal = merge_cls_sal
        self.args = args
        self.feat_interp = getattr(args, "feat_interp", 1)
        self._eff_clip = args.clip_length / self.feat_interp  # effective clip length after interpolation
        self.x = nn.Parameter(torch.tensor(0.5))
        self.use_post_verification = bool(getattr(args, "use_SRM", False))
        self.use_pv_repr = self.use_post_verification and getattr(args, "use_pv_repr", False)
        self.use_pv_adj = self.use_post_verification and getattr(args, "use_pv_adj", False)
        if self.use_post_verification:
            # Experimental paper-consistent post-verification module.
            self.pv_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.pv_head = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3 + 1),
                nn.Linear(hidden_dim * 3 + 1, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
        if self.use_pv_adj:
            self.pv_adj_head = nn.Linear(hidden_dim, 2)


    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, vid, qid, targets=None):
        # Project inputs to the same hidden dimension
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        # Add type embeddings
        src_vid = src_vid + self.token_type_embeddings(torch.full_like(src_vid_mask.long(), 1))
        src_txt = src_txt + self.token_type_embeddings(torch.zeros_like(src_txt_mask.long()))
        # Add position embeddings
        pos_vid = self.position_embed(src_vid, src_vid_mask)
        if self.use_txt_pos:
            pos_txt = self.txt_position_embed(src_txt)
        else:
            pos_txt = torch.zeros_like(src_txt, device=src_txt.device)

        # Insert dummy tokens in front of text
        batch_size = src_txt.shape[0]
        txt_dummy = self.dummy_rep_token.unsqueeze(0).expand(batch_size, -1, -1)
        pos_dummy = self.dummy_rep_pos.unsqueeze(0).expand(batch_size, -1, -1)
        mask_txt = torch.ones(
            (batch_size, self.args.num_dummies),
            dtype=torch.bool,
            device=src_txt_mask.device,
        )

        src_txt_dummy = torch.cat([txt_dummy, src_txt], dim=1)
        src_txt_mask_dummy = torch.cat([mask_txt, src_txt_mask], dim=1)
        pos_txt_dummy = torch.cat([pos_dummy, pos_txt], dim=1)

        src_txt_dummy = src_txt_dummy.permute(1, 0, 2)
        pos_txt_dummy = pos_txt_dummy.permute(1, 0, 2)

        memory = self.txtproj_encoder(
            src_txt_dummy,
            src_key_padding_mask=~(src_txt_mask_dummy.bool()),
            pos=pos_txt_dummy,
        )
        dummy_token = memory[: self.args.num_dummies].permute(1, 0, 2)
        query_emb = dummy_token.mean(dim=1, keepdim=True)
        pos_txt_dummy = pos_txt_dummy.permute(1, 0, 2)

        src_txt_dummy = torch.cat([dummy_token, src_txt], dim=1)
        mask_txt_dummy = torch.tensor([[True] * self.args.num_dummies], device=src_txt_mask.device).repeat(
            src_txt_mask.shape[0], 1
        )
        src_txt_mask_dummy = torch.cat([mask_txt_dummy, src_txt_mask], dim=1)

        src = torch.cat([src_vid, src_txt_dummy], dim=1)
        mask = torch.cat([src_vid_mask, src_txt_mask_dummy], dim=1).bool()
        pos = torch.cat([pos_vid, pos_txt_dummy], dim=1)

        video_length = src_vid.shape[1]

        video_emb, video_msk, _, _, saliency_scores = self.transformer(
            src,
            ~mask,
            pos,
            video_length=video_length,
            saliency_proj1=self.saliency_proj1,
            saliency_proj2=self.saliency_proj2,
        )

        video_emb = video_emb.permute(1, 0, 2)
        video_msk = (~video_msk).int()

        if self.feat_interp > 1:
            B, T, C = video_emb.shape
            new_T = T * self.feat_interp
            video_emb = F.interpolate(
                video_emb.transpose(1, 2), size=new_T, mode="linear", align_corners=False
            ).transpose(1, 2)
            video_msk = F.interpolate(
                video_msk.float().unsqueeze(1), size=new_T, mode="nearest"
            ).squeeze(1).long()
            if self.training:
                targets["saliency_all_labels"] = F.interpolate(
                    targets["saliency_all_labels"].float().unsqueeze(1),
                    size=new_T, mode="nearest"
                ).squeeze(1).long()
                targets["saliency_pos_labels"] = F.interpolate(
                    targets["saliency_pos_labels"].float().unsqueeze(1),
                    size=new_T, mode="nearest"
                ).squeeze(1).long()

        pymid, pymid_msk = self.pyramid(video_emb, video_msk, return_mask=self.training)
        point = self.generator(pymid)

        with torch.autocast("cuda", enabled=False):
            video_emb = video_emb.float()
            out_class = [self.class_head(e.float()) for e in pymid]
            out_class = torch.cat(out_class, dim=1)
            out_conf = torch.cat(pymid, dim=1)
            out_conf = self.conf_head(out_conf)
            out_class = self.x * out_class + (1 - self.x) * out_conf

            out_coord = None
            if self.coord_head is not None and len(pymid) > 0:
                out_coord = [
                    self.coord_head(e.float()).exp() * self.coef[i]
                    for i, e in enumerate(pymid)
                ]
                out_coord = torch.cat(out_coord, dim=1)

        if out_coord is None:
            raise RuntimeError("coord_head did not produce localization results; inference cannot proceed.")

        output = dict(saliency_scores=saliency_scores)

        if self.training:
            if self.criterion is None:
                raise RuntimeError("loss_cfg is required for training")
            if targets is None:
                raise RuntimeError("targets are required for training")

            boundaries = self._target_boundaries_to_seconds(
                targets["span_labels"], src_vid_mask, src_vid.device
            )
            loss_data = dict(
                boundary=boundaries,
                fps=torch.full(
                    (src_vid.size(0),),
                    1 / self.args.clip_length,
                    dtype=video_emb.dtype,
                    device=src_vid.device,
                ),
                point=point,
                out_class=out_class,
                out_coord=out_coord,
                pymid_msk=pymid_msk,
                video_emb=video_emb,
                query_emb=query_emb.expand(-1, video_emb.size(1), -1),
                video_msk=video_msk,
                saliency=targets["saliency_all_labels"],
                pos_clip=targets["saliency_pos_labels"],
            )
            output = self.criterion(loss_data, output)
            if self.use_post_verification:
                proposal_msk = torch.cat(pymid_msk, dim=1).bool()
                loss_pv, pv_logits, top_scores, selected_boundaries, pooled = self._post_verification_loss(
                    out_class, out_coord, point, video_emb, video_msk, query_emb, boundaries, proposal_msk
                )
                output["loss_pv"] = loss_pv
                extra = self._compute_pv_extra_losses(
                    pv_logits, top_scores, selected_boundaries, pooled, boundaries
                )
                output.update(extra)
            return output

        bs = src_vid.shape[0]
        if bs != 1:
            raise AssertionError("batch size larger than 1 is not supported for inference")

        out_class = out_class.sigmoid()

        boundaries = self._decode_boundaries(out_coord, point)
        # Inference: paper-consistent — raw detection head output.
        # PV module is training-only auxiliary supervision (L_PV + L_repr).
        boundary = torch.cat((boundaries[0], out_class[0]), dim=-1)
        _, inds = out_class[0, :, 0].sort(descending=True)
        boundary = boundary[inds[: self.max_num_moment]]

        output["_out"] = dict(
            label=None if targets is None else targets.get("label", [None])[0],
            video_msk=video_msk,
            saliency=saliency_scores[0],
            boundary=boundary,
        )

        return output

    def _decode_boundaries(self, out_coord, point):
        boundaries = out_coord.clone()
        boundaries[:, :, 0] *= -1
        stride = point[:, 3].to(device=out_coord.device, dtype=out_coord.dtype).view(1, -1, 1)
        center = point[:, 0].to(device=out_coord.device, dtype=out_coord.dtype).view(1, -1, 1)
        boundaries = boundaries * stride + center
        boundaries = boundaries / (1 / self._eff_clip)
        start = torch.minimum(boundaries[:, :, 0], boundaries[:, :, 1])
        end = torch.maximum(boundaries[:, :, 0], boundaries[:, :, 1])
        return torch.stack((start, end), dim=-1)

    def _select_top_proposals(self, base_scores, proposal_msk=None):
        if proposal_msk is not None:
            valid_counts = proposal_msk.sum(dim=1)
            if torch.any(valid_counts == 0):
                raise RuntimeError("post-verification received a sample with no valid proposals")
            k = min(self.max_num_moment, int(valid_counts.min().item()), base_scores.size(1))
            base_scores = base_scores.masked_fill(~proposal_msk, torch.finfo(base_scores.dtype).min)
        else:
            k = min(self.max_num_moment, base_scores.size(1))
        scores, indices = base_scores.topk(k, dim=1)
        return scores, indices

    def _gather_boundaries(self, boundaries, indices):
        return boundaries.gather(1, indices.unsqueeze(-1).expand(-1, -1, 2))

    def _pool_moment_representations(self, video_emb, video_msk, boundaries):
        batch_size, num_clips, hidden_dim = video_emb.shape
        clip_centers = (
            torch.arange(num_clips, device=video_emb.device, dtype=video_emb.dtype) + 0.5
        ) * self._eff_clip
        pooled = video_emb.new_zeros(batch_size, boundaries.size(1), hidden_dim)

        for b in range(batch_size):
            valid = video_msk[b].bool()
            valid_centers = clip_centers[valid]
            valid_emb = video_emb[b, valid]
            if valid_emb.numel() == 0:
                valid_centers = clip_centers
                valid_emb = video_emb[b]
            for k in range(boundaries.size(1)):
                start, end = boundaries[b, k]
                inside = (valid_centers >= start) & (valid_centers <= end)
                if inside.any():
                    pooled[b, k] = valid_emb[inside].mean(dim=0)
                else:
                    center = (start + end) * 0.5
                    nearest = (valid_centers - center).abs().argmin()
                    pooled[b, k] = valid_emb[nearest]

        return pooled

    def _sort_by_start_time(self, boundaries, scores):
        """Sort proposals by start time for temporal-order GRU processing."""
        start = boundaries[:, :, 0]  # (B, K)
        _, sort_idx = start.sort(dim=1)  # ascending by start time
        sorted_boundaries = boundaries.gather(1, sort_idx.unsqueeze(-1).expand(-1, -1, 2))
        sorted_scores = scores.gather(1, sort_idx)
        return sorted_boundaries, sorted_scores

    def _post_verification_logits(self, pooled, query_emb, base_scores):
        gru_out, _ = self.pv_gru(pooled)
        query = query_emb.expand(-1, pooled.size(1), -1)
        head_in = torch.cat((gru_out, pooled, query, base_scores.unsqueeze(-1)), dim=-1)
        return self.pv_head(head_in).squeeze(-1)

    def _post_verification_loss(self, out_class, out_coord, point, video_emb, video_msk, query_emb, gt_boundaries, proposal_msk):
        boundaries = self._decode_boundaries(out_coord, point)
        base_scores = out_class.sigmoid().squeeze(-1)
        top_scores, top_indices = self._select_top_proposals(base_scores, proposal_msk)
        selected_boundaries = self._gather_boundaries(boundaries, top_indices)
        selected_boundaries, top_scores = self._sort_by_start_time(selected_boundaries, top_scores)
        pooled = self._pool_moment_representations(video_emb, video_msk, selected_boundaries)
        pv_logits = self._post_verification_logits(pooled, query_emb, top_scores)
        targets = self._best_tiou_targets(selected_boundaries, gt_boundaries)
        pv_scores = pv_logits.sigmoid()
        loss_pv = F.mse_loss(pv_scores, targets)
        return loss_pv, pv_logits, top_scores, selected_boundaries, pooled

    def _compute_pv_extra_losses(self, pv_logits, top_scores, selected_boundaries, pooled, gt_boundaries):
        extra = {}
        # L_repr: cross-moment representation consistency loss
        if self.use_pv_repr:
            pooled_n = F.normalize(pooled, dim=-1)
            S = torch.bmm(pooled_n, pooled_n.transpose(1, 2)).clamp(-1, 1)
            S_pos = (S + 1) / 2  # map cosine [-1,1] to [0,1]
            # Target T[i,j] = 1 if proposals i and j best-match the same GT
            # ("agreement from the ground truth", per paper)
            K = selected_boundaries.size(1)
            T = selected_boundaries.new_zeros(selected_boundaries.size(0), K, K)
            for b in range(selected_boundaries.size(0)):
                sb = selected_boundaries[b]  # (K, 2)
                gt = gt_boundaries[b].to(device=sb.device, dtype=sb.dtype)
                if gt.numel() == 0:
                    T[b].fill_diagonal_(0)
                    continue
                # proposal-GT tIoU → (K, G)
                inter_start = torch.maximum(sb[:, None, 0], gt[None, :, 0])
                inter_end = torch.minimum(sb[:, None, 1], gt[None, :, 1])
                inter = (inter_end - inter_start).clamp(min=0)
                prop_len = (sb[:, 1] - sb[:, 0]).clamp(min=1e-6)
                gt_len = (gt[:, 1] - gt[:, 0]).clamp(min=1e-6)
                union = prop_len[:, None] + gt_len[None, :] - inter
                tiou = inter / union.clamp(min=1e-6)
                # best GT per proposal; unmatched → unique negative ID
                best_tiou, best_gt = tiou.max(dim=1)
                unmatched = best_tiou <= 0
                if unmatched.any():
                    neg_ids = -1 - torch.arange(
                        unmatched.sum(), device=best_gt.device, dtype=best_gt.dtype
                    )
                    best_gt = best_gt.masked_scatter(unmatched, neg_ids)
                T[b] = (best_gt[:, None] == best_gt[None, :]).float()
                T[b].fill_diagonal_(0)  # exclude self
            # CE loss per paper: L_repr = CE(S, T), exclude self-pairs
            mask = 1 - torch.eye(K, device=T.device).unsqueeze(0).expand(T.size(0), -1, -1)
            S_pos = S_pos.clamp(1e-7, 1 - 1e-7)
            extra["loss_pv_repr"] = F.binary_cross_entropy(
                S_pos.flatten(1), T.flatten(1), weight=mask.flatten(1)
            )
        # L_adj: boundary adjustment loss
        if self.use_pv_adj:
            deltas = self.pv_adj_head(pooled)  # (B, K, 2): delta_start, delta_end
            refined = selected_boundaries + deltas * self._eff_clip
            tiou_targets = self._best_tiou_targets(refined, gt_boundaries)
            pv_scores_refined = pv_logits.sigmoid()
            extra["loss_pv_adj"] = F.mse_loss(pv_scores_refined, tiou_targets)
        return extra

    def _rerank_with_post_verification(self, boundaries, base_scores, video_emb, video_msk, query_emb):
        top_scores, top_indices = self._select_top_proposals(base_scores)
        selected_boundaries = self._gather_boundaries(boundaries, top_indices)
        selected_boundaries, top_scores = self._sort_by_start_time(selected_boundaries, top_scores)
        pooled = self._pool_moment_representations(video_emb, video_msk, selected_boundaries)
        pv_scores = self._post_verification_logits(pooled, query_emb, top_scores).sigmoid()
        refined_scores = top_scores * pv_scores
        if self.use_pv_adj:
            deltas = self.pv_adj_head(pooled)  # (1, K, 2)
            selected_boundaries = selected_boundaries + deltas * self._eff_clip
        boundary = torch.cat((selected_boundaries, refined_scores.unsqueeze(-1)), dim=-1)
        order = refined_scores.argsort(dim=1, descending=True)
        return boundary.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))

    def _best_tiou_targets(self, proposals, gt_boundaries):
        targets = proposals.new_zeros(proposals.shape[:2])
        for b, gt in enumerate(gt_boundaries):
            gt = gt.to(device=proposals.device, dtype=proposals.dtype)
            if gt.numel() == 0:
                continue
            prop = proposals[b]
            inter_start = torch.maximum(prop[:, None, 0], gt[None, :, 0])
            inter_end = torch.minimum(prop[:, None, 1], gt[None, :, 1])
            inter = (inter_end - inter_start).clamp(min=0)
            prop_len = (prop[:, 1] - prop[:, 0]).clamp(min=0)
            gt_len = (gt[:, 1] - gt[:, 0]).clamp(min=0)
            union = prop_len[:, None] + gt_len[None, :] - inter
            tiou = inter / union.clamp(min=1e-6)
            targets[b] = tiou.max(dim=1).values
        return targets

    def _target_boundaries_to_seconds(self, span_targets, src_vid_mask, device):
        boundaries = []
        valid_lengths = src_vid_mask.sum(dim=1).to(device=device, dtype=torch.float32)

        for i, target in enumerate(span_targets):
            spans = target["spans"].to(device)
            if self.args.span_loss_type == "l1":
                xx = span_cxw_to_xx(spans.float())
                seconds = xx * (valid_lengths[i] * self.args.clip_length)
            elif self.args.span_loss_type == "ce":
                seconds = spans.float()
                seconds[:, 1] += 1
                seconds = seconds * self.args.clip_length
            else:
                raise NotImplementedError(f"Unsupported span_loss_type: {self.args.span_loss_type}")
            seconds[:, 0] = seconds[:, 0].clamp(min=0)
            seconds[:, 1] = torch.maximum(seconds[:, 1], seconds[:, 0] + 1e-4)
            boundaries.append(seconds)

        return boundaries



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, input_dim, output_dim, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(input_dim)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = FlashMMR(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        input_dropout=args.input_dropout,
        n_input_proj=args.n_input_proj,
        strides=args.cfg.model.strides,
        buffer_size=args.cfg.model.buffer_size,
        max_num_moment=args.cfg.model.max_num_moment,
        pyramid_cfg=args.cfg.model.pyramid_cfg,
        coord_head_cfg=args.cfg.model.coord_head_cfg,
        loss_cfg=args.cfg.model.loss_cfg,
        args=args
    )

    return model
