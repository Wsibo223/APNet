#!/usr/bin/env python3
"""P16: capacity-scaled multi-scale symptom-dynamics network.

The model has one shared 71-D pose-token stem. Six small temporal basis
modules operate at different receptive fields before information is pooled.
Each basis owns a symptom-guided evidence query and score head. A learned
sample-wise router consumes the evidence and head disagreement and makes one
straight-through hard choice. There is no external teacher or P2/P3/P5 input.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn
import torch.nn.functional as F


EVIDENCE_NAMES = ("global", "stillness", "burst", "retardation", "topk", "severe")


def masked_mean(x, mask):
    weight = mask.float().unsqueeze(-1)
    return (x * weight).sum(1) / weight.sum(1).clamp_min(1.)


def masked_max(x, mask):
    return x.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values


class TemporalBlock(nn.Module):
    def __init__(self, dim, kernel, dilation, dropout):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.depthwise = nn.Conv1d(
            dim, dim, kernel, padding=padding, dilation=dilation, groups=dim)
        self.pointwise = nn.Conv1d(dim, 2 * dim, 1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * dim, dim))

    def forward(self, x, mask):
        value, gate = self.pointwise(
            self.depthwise(x.transpose(1, 2))).chunk(2, dim=1)
        value = (value * torch.sigmoid(gate)).transpose(1, 2)
        x = self.norm1(x + self.dropout(value))
        x = x * mask.float().unsqueeze(-1)
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x * mask.float().unsqueeze(-1)


class Projection(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim),
            nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(output_dim))

    def forward(self, x):
        return self.net(x)


class GuidedSymptomPool(nn.Module):
    def __init__(self, dim, evidence_dim, dropout):
        super().__init__()
        hidden = max(12, dim // 2)
        self.score = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.readout = nn.Sequential(
            nn.LayerNorm(3 * dim + 3), nn.Linear(3 * dim + 3, evidence_dim),
            nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(evidence_dim))

    def forward(self, x, mask, prior, guide_strength):
        logits = self.score(x).squeeze(-1)
        logits = logits + float(guide_strength) * torch.log(prior.clamp_min(1e-4))
        logits = logits.masked_fill(~mask, -1e4)
        attention = torch.softmax(logits, 1) * mask.float()
        attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("bt,btd->bd", attention, x)
        mean = masked_mean(x, mask)
        maximum = masked_max(x, mask)
        strength = (attention * prior).sum(1)
        peak = attention.max(1).values
        entropy = -(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()).sum(1)
        entropy = entropy / mask.float().sum(1).clamp_min(2.).log()
        evidence = self.readout(torch.cat([
            pooled, mean, maximum, strength[:, None], peak[:, None], entropy[:, None]], 1))
        return evidence, attention, strength, peak, entropy


class SymptomScale(nn.Module):
    def __init__(self, dim, evidence_dim, kernel, dilation, depth, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            TemporalBlock(dim, kernel, dilation, dropout) for _ in range(depth)])
        self.pool = GuidedSymptomPool(dim, evidence_dim, dropout)

    def forward(self, x, mask, prior, guide_strength):
        for block in self.blocks:
            x = block(x, mask)
        return self.pool(x, mask, prior, guide_strength)


class MultiScaleSymptomPRE(nn.Module):
    """One model; shared token stem; six internal temporal evidence scales."""

    def __init__(self, dimensions: Dict[str, int], token_dim=71, model_dim=48,
                 evidence_dim=24, context_dim=64, stem_depth=1, branch_depth=1,
                 kernels: Sequence[int] = (3, 7, 3, 5, 3, 7),
                 dilations: Sequence[int] = (1, 1, 1, 2, 4, 4),
                 dropout=.12, guide_strength=3., local_cap=36., base_weight=0.,
                 router_temperature=.7, gain_cap=1.75, bias_cap=8.,
                 route_edges: Sequence[float] = (4., 6., 8., 11., 15.),
                 score_min=0., score_max=63.):
        super().__init__()
        if len(kernels) != len(EVIDENCE_NAMES) or len(dilations) != len(EVIDENCE_NAMES):
            raise ValueError("P16 requires one temporal scale per symptom head")
        self.n_heads = len(EVIDENCE_NAMES)
        self.base_weight = float(base_weight)
        self.local_cap = float(local_cap)
        self.router_temperature = float(router_temperature)
        self.gain_cap = float(gain_cap)
        self.bias_cap = float(bias_cap)
        self.guide_strength = float(guide_strength)
        self.score_min, self.score_max = float(score_min), float(score_max)
        self.register_buffer("route_edges", torch.as_tensor(route_edges, dtype=torch.float32))

        self.token_projection = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, model_dim), nn.GELU(),
            nn.Dropout(dropout))
        self.shared_stem = nn.ModuleList([
            TemporalBlock(model_dim, 3, 1, dropout) for _ in range(int(stem_depth))])
        self.scales = nn.ModuleList([
            SymptomScale(model_dim, evidence_dim, int(kernel), int(dilation),
                         int(branch_depth), dropout)
            for kernel, dilation in zip(kernels, dilations)])

        aee_input = dimensions["aee_ff"] * 4 + dimensions["aee_evidence"]
        global_input = (dimensions["salient_evidence"] + dimensions["severe_evidence"]
                        + dimensions["score_context"] + dimensions["quality_context"])
        self.aee_projection = Projection(aee_input, context_dim, dropout)
        self.global_projection = Projection(global_input, context_dim, dropout)
        role_input = evidence_dim * 5 + 2
        self.role_projection = nn.ModuleList([
            Projection(role_input, context_dim, dropout) for _ in EVIDENCE_NAMES])
        self.head_fusion = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(context_dim * 3), nn.Linear(context_dim * 3, context_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(context_dim, context_dim),
                nn.GELU(), nn.LayerNorm(context_dim))
            for _ in EVIDENCE_NAMES])
        self.head_regressor = nn.ModuleList([
            nn.Sequential(nn.Linear(context_dim, context_dim), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(context_dim, 1))
            for _ in EVIDENCE_NAMES])

        router_input = context_dim * 2 + self.n_heads * 3 + 3
        self.router = nn.Sequential(
            nn.LayerNorm(router_input), nn.Linear(router_input, context_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(context_dim, context_dim), nn.GELU(),
            nn.Linear(context_dim, self.n_heads))
        self.gain_bias = nn.Sequential(
            nn.LayerNorm(router_input), nn.Linear(router_input, context_dim // 2), nn.GELU(),
            nn.Linear(context_dim // 2, 2))
        self.severe_classifier = nn.Sequential(
            nn.LayerNorm(evidence_dim * 3), nn.Linear(evidence_dim * 3, evidence_dim),
            nn.GELU(), nn.Linear(evidence_dim, 1))

    def _encode_view(self, tokens, mask, priors):
        x = self.token_projection(tokens) * mask.float().unsqueeze(-1)
        for block in self.shared_stem:
            x = block(x, mask)
        values = [scale(x, mask, priors[:, :, index], self.guide_strength)
                  for index, scale in enumerate(self.scales)]
        evidence = torch.stack([item[0] for item in values], 1)
        attention = torch.stack([item[1] for item in values], 2)
        strength = torch.stack([item[2] for item in values], 1)
        peak = torch.stack([item[3] for item in values], 1)
        entropy = torch.stack([item[4] for item in values], 1)
        return evidence, attention, strength, peak, entropy

    def forward(self, groups, ff_tokens, ff_mask, ff_priors,
                nw_tokens, nw_mask, nw_priors, aee_pred, pre_pred):
        ff, ff_attn, ff_strength, ff_peak, ff_entropy = self._encode_view(
            ff_tokens, ff_mask, ff_priors)
        nw, nw_attn, nw_strength, nw_peak, nw_entropy = self._encode_view(
            nw_tokens, nw_mask, nw_priors)
        mean = .5 * (ff + nw)
        gap = torch.abs(ff - nw)
        product = ff * nw
        pair_strength = .5 * (ff_strength + nw_strength)
        strength_gap = torch.abs(ff_strength - nw_strength)
        role_features = torch.cat([
            ff, nw, mean, gap, product, pair_strength.unsqueeze(-1),
            strength_gap.unsqueeze(-1)], -1)

        aee_ff, aee_nw = groups["aee_ff"], groups["aee_nw"]
        aee_context = self.aee_projection(torch.cat([
            aee_ff, aee_nw, .5 * (aee_ff + aee_nw), torch.abs(aee_ff - aee_nw),
            groups["aee_evidence"]], 1))
        global_context = self.global_projection(torch.cat([
            groups["salient_evidence"], groups["severe_evidence"],
            groups["score_context"], groups["quality_context"]], 1))
        base = (1. - self.base_weight) * aee_pred + self.base_weight * pre_pred

        head_predictions, head_residuals = [], []
        for index in range(self.n_heads):
            role = self.role_projection[index](role_features[:, index])
            context = self.head_fusion[index](torch.cat([
                role, aee_context, global_context], 1))
            residual = self.local_cap * torch.tanh(
                self.head_regressor[index](context).squeeze(-1))
            prediction = (base + residual).clamp(self.score_min, self.score_max)
            head_predictions.append(prediction)
            head_residuals.append(prediction - base)
        head_prediction = torch.stack(head_predictions, 1)
        head_residual = torch.stack(head_residuals, 1)
        disagreement_range = head_prediction.max(1).values - head_prediction.min(1).values
        disagreement_std = head_prediction.std(1, unbiased=False)

        router_features = torch.cat([
            aee_context, global_context, head_residual / self.local_cap,
            pair_strength, strength_gap,
            disagreement_range[:, None] / self.local_cap,
            disagreement_std[:, None] / self.local_cap,
            (aee_pred - pre_pred)[:, None] / 32.], 1)
        router_logits = self.router(router_features)
        router_soft = torch.softmax(
            router_logits / max(self.router_temperature, 1e-3), 1)
        router_hard = F.one_hot(router_soft.argmax(1), self.n_heads).to(router_soft.dtype)
        selector = router_hard + router_soft - router_soft.detach()
        reference_residual = (selector * head_residual).sum(1)
        raw_gain_bias = self.gain_bias(router_features)
        gain = .25 + (self.gain_cap - .25) * torch.sigmoid(raw_gain_bias[:, 0])
        bias = self.bias_cap * torch.tanh(raw_gain_bias[:, 1])
        prediction = (base + gain * reference_residual + bias).clamp(
            self.score_min, self.score_max)

        severe_index = EVIDENCE_NAMES.index("severe")
        sev = torch.cat([ff[:, severe_index], nw[:, severe_index], mean[:, severe_index]], 1)
        severe_logit = self.severe_classifier(sev).squeeze(-1)
        route_bin = torch.bucketize(disagreement_range.detach(), self.route_edges)
        return {
            "pred": prediction, "base": base, "residual": prediction - base,
            "head_pred": head_prediction, "head_residual": head_residual,
            "disagreement_range": disagreement_range,
            "disagreement_std": disagreement_std, "route_bin": route_bin,
            "route_selector": router_hard, "route_selector_soft": router_soft,
            "router_logits": router_logits,
            "route_reference": base + reference_residual,
            "route_alpha_base": torch.ones_like(gain),
            "route_alpha_reference": gain, "route_bias": bias,
            "severe_logit": severe_logit,
            "evidence_strength": pair_strength,
            "view_strength_gap": strength_gap,
            "evidence_peak": .5 * (ff_peak + nw_peak),
            "evidence_entropy": .5 * (ff_entropy + nw_entropy),
            "ff_attention": ff_attn, "nw_attention": nw_attn,
            "ff_priors": ff_priors, "nw_priors": nw_priors,
        }


def ccc_loss(pred, target):
    covariance = ((pred - pred.mean()) * (target - target.mean())).mean()
    denominator = (pred.var(unbiased=False) + target.var(unbiased=False)
                   + (pred.mean() - target.mean()).square() + 1e-8)
    return 1. - 2. * covariance / denominator


def rank_loss(pred, target, margin):
    td = target[:, None] - target[None, :]
    pd = pred[:, None] - pred[None, :]
    valid = td.abs() >= 3.
    return F.relu(margin - torch.sign(td) * pd)[valid].mean() if valid.any() \
        else pred.new_tensor(0.)


def attention_diversity(attention, mask):
    value = attention * mask.float().unsqueeze(-1)
    value = value / value.square().sum(1, keepdim=True).sqrt().clamp_min(1e-6)
    gram = torch.einsum("btk,btl->bkl", value, value)
    eye = torch.eye(gram.shape[-1], device=gram.device).unsqueeze(0)
    return ((gram - eye) * (1. - eye)).square().mean()


def head_role_weights(output, target, config):
    strength = output["evidence_strength"].detach()
    weights = .5 + strength
    weights[:, 0] = 1.
    weights[:, 1] *= 1. + .5 * strength[:, 1]
    weights[:, 2] *= 1. + .5 * strength[:, 2]
    weights[:, 3] *= 1. + .75 * (target >= 14.).float()
    weights[:, 4] *= 1. + .5 * (torch.abs(target - output["base"].detach()) >= 8.).float()
    weights[:, 5] *= 1. + float(config["severe_weight"]) * (target >= 29.).float()
    return weights / weights.mean(0, keepdim=True).clamp_min(1e-6)


def p16_loss(output, target, config):
    pred = output["pred"]
    huber = F.smooth_l1_loss(pred, target, beta=float(config["huber_beta"]))
    mse = F.mse_loss(pred, target)
    ccc = ccc_loss(pred, target)
    ranking = rank_loss(pred, target, float(config["rank_margin"]))
    spread = ((pred.std(unbiased=False) - target.std(unbiased=False))
              / target.std(unbiased=False).clamp_min(1e-6)).square()

    head_abs = torch.abs(output["head_pred"] - target[:, None])
    head_huber = F.smooth_l1_loss(
        output["head_pred"], target[:, None].expand_as(output["head_pred"]),
        beta=float(config["huber_beta"]), reduction="none")
    head = (head_huber * head_role_weights(output, target, config)).mean()
    head_ccc = torch.stack([
        ccc_loss(output["head_pred"][:, i], target)
        for i in range(output["head_pred"].shape[1])]).mean()

    oracle_target = torch.softmax(
        -head_abs.detach() / float(config["oracle_temperature"]), 1)
    router_log = torch.log_softmax(
        output["router_logits"] / max(float(config["router_temperature"]), 1e-3), 1)
    router_oracle = -(oracle_target * router_log).sum(1).mean()
    routed_expected_error = (output["route_selector_soft"] * head_abs).sum(1).mean()

    needed = torch.abs(target - output["base"]).detach().clamp(
        max=float(config["range_cap"]))
    range_calibration = F.smooth_l1_loss(
        output["disagreement_range"], needed, beta=2.)
    severe_target = (target >= 29.).float()
    positive = severe_target.sum().clamp_min(1.)
    negative = (1. - severe_target).sum().clamp_min(1.)
    severe = F.binary_cross_entropy_with_logits(
        output["severe_logit"], severe_target,
        pos_weight=(negative / positive).detach())
    ordinal = pred.new_tensor(0.)
    for threshold in (14., 21., 29.):
        ordinal = ordinal + F.binary_cross_entropy_with_logits(
            (pred - threshold) / float(config["ordinal_temperature"]),
            (target >= threshold).float())
    ordinal = ordinal / 3.
    attention = .5 * (
        attention_diversity(output["ff_attention"], output["ff_attention"].sum(-1) > 0)
        + attention_diversity(output["nw_attention"], output["nw_attention"].sum(-1) > 0))
    high_under = (F.relu(target - pred) * (target >= 29.).float()).square().mean()
    low_over = (F.relu(pred - target) * (target <= 13.).float()).square().mean()
    asymmetric = high_under + .5 * low_over
    router_use = output["route_selector_soft"].mean(0)
    router_balance = (router_use * router_use.clamp_min(1e-8).log()).sum()

    total = (huber + float(config["w_mse"]) * mse
             + float(config["w_ccc"]) * ccc
             + float(config["w_rank"]) * ranking
             + float(config["w_spread"]) * spread
             + float(config["w_head"]) * head
             + float(config["w_head_ccc"]) * head_ccc
             + float(config["w_router_oracle"]) * router_oracle
             + float(config["w_routed_error"]) * routed_expected_error
             + float(config["w_range"]) * range_calibration
             + float(config["w_severe"]) * severe
             + float(config["w_ordinal"]) * ordinal
             + float(config["w_attention_diversity"]) * attention
             + float(config["w_asymmetric"]) * asymmetric
             + float(config["w_router_balance"]) * router_balance)
    return total, {
        "loss": total, "huber": huber, "mse": mse, "ccc": ccc,
        "rank": ranking, "spread": spread, "head": head,
        "head_ccc": head_ccc, "router_oracle": router_oracle,
        "routed_expected_error": routed_expected_error,
        "range_calibration": range_calibration, "severe": severe,
        "ordinal": ordinal, "attention_diversity": attention,
        "asymmetric": asymmetric, "router_balance": router_balance,
    }
