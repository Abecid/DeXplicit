# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn.functional as F
from torch.nn import HuberLoss
from pointworld import metrics as metrics_utils
from utils import safe_loss_computation


def compute_single_output_loss(
    args,
    output_norm,
    gt_target_norm,
    weights,
    pred_exists_supervised,
    log_var,
    var_floor: float,
    var_ceiling: float,
    uncertainty_logvar_weight: float,
    moved_mask=None,
    static_mask=None,
    pred_relative=None,
    gt_relative=None,
    selector_gt=None,
):
    # Standard continuous prediction loss
    huber = HuberLoss(delta=args.huber_delta, reduction="none")
    error_term = huber(output_norm, gt_target_norm)  # (B,T,NS,3)
    per_point_loss = error_term.mean(dim=-1)  # (B,T,NS)

    # Apply uncertainty re-weighting (always on)
    with torch.autocast("cuda", enabled=False):
        # Clamp log-variance directly to keep NLL bounded
        s_min = math.log(var_floor)
        s_max = math.log(var_ceiling)
        if log_var.shape[-1] == 1:
            log_var_expanded = log_var.expand_as(error_term)
        else:
            log_var_expanded = log_var

        log_var_clamped = log_var_expanded.clamp(min=s_min, max=s_max)
        var = torch.exp(log_var_clamped)  # (B,T,NS,1|3)

        per_dim_loss = 0.5 * (
            error_term / var +
            uncertainty_logvar_weight * log_var_clamped
        )
        per_point_loss = per_dim_loss.mean(dim=-1)  # (B,T,NS)

    def _masked_mean(tensor, mask):
        mask = mask.to(dtype=torch.bool, device=tensor.device)
        if mask.any():
            return tensor[mask].mean()
        return tensor.sum() * 0.0

    def _weighted_mean(tensor, item_weights):
        denom = item_weights.sum()
        if denom > 0:
            return (tensor * item_weights).sum() / denom
        return tensor.sum() * 0.0

    loss_terms = {}
    valid = pred_exists_supervised.to(dtype=torch.bool, device=per_point_loss.device)

    if bool(getattr(args, "balanced_moved_static_loss", False)):
        if selector_gt is None or static_mask is None:
            raise ValueError("balanced_moved_static_loss requires selector_gt and static_mask.")
        selector = selector_gt.to(dtype=per_point_loss.dtype, device=per_point_loss.device).clamp(0.0, 1.0)
        moved_weights = torch.where(valid, selector, torch.zeros_like(selector))
        static_valid = valid & static_mask.to(dtype=torch.bool, device=per_point_loss.device)

        moved_regression_loss = _weighted_mean(per_point_loss, moved_weights)
        static_regression_loss = _masked_mean(per_point_loss, static_valid)
        dynamics_loss = (
            float(getattr(args, "moved_loss_weight", 1.0)) * moved_regression_loss
            + float(getattr(args, "static_loss_weight", 1.0)) * static_regression_loss
        )
        loss_terms["moved_regression_loss"] = moved_regression_loss
        loss_terms["static_regression_loss"] = static_regression_loss
        loss_terms["balanced_moved_weight_sum"] = moved_weights.sum()
        loss_terms["balanced_static_count"] = static_valid.sum()
    else:
        # Apply legacy selector weights and sum
        dynamics_loss = (per_point_loss * weights).sum()

    total_loss = dynamics_loss

    static_disp_weight = float(getattr(args, "static_displacement_loss_weight", 0.0))
    if static_disp_weight > 0.0:
        if pred_relative is None:
            raise ValueError("static_displacement_loss_weight > 0 requires pred_relative.")
        if gt_relative is None:
            raise ValueError("static_displacement_loss_weight > 0 requires gt_relative.")
        static_tau = float(getattr(args, "static_displacement_tau", 0.005))
        cumulative_static = torch.norm(gt_relative, dim=-1) <= static_tau
        static_valid = valid & cumulative_static.to(dtype=torch.bool, device=per_point_loss.device)
        static_displacement_loss = _masked_mean(torch.norm(pred_relative, dim=-1), static_valid)
        total_loss = total_loss + static_disp_weight * static_displacement_loss
        loss_terms["static_displacement_loss"] = static_displacement_loss

    loss_terms["dynamics_loss"] = dynamics_loss

    return total_loss, per_point_loss, loss_terms


def compute_move_prediction_loss(args, outputs, gt_relative, pred_exists_supervised):
    if "move_logits" not in outputs:
        raise ValueError("use_move_prediction_head=true but model outputs are missing move_logits.")

    move_logits = outputs["move_logits"]
    valid = pred_exists_supervised.to(dtype=torch.bool, device=move_logits.device)
    gt_relative = gt_relative.to(device=move_logits.device)

    tau = float(getattr(args, "move_prediction_tau", 0.005))
    temp_scale = float(getattr(args, "move_prediction_temp_scale", 5.0))
    k = temp_scale / max(tau, 1e-12)
    gt_motion_norm = torch.norm(gt_relative, dim=-1)
    move_targets = torch.sigmoid(k * (gt_motion_norm - tau)).to(dtype=move_logits.dtype)

    per_point_bce = F.binary_cross_entropy_with_logits(
        move_logits,
        move_targets,
        reduction="none",
    )

    hard_moved = move_targets > 0.5
    moved_valid = valid & hard_moved
    static_valid = valid & ~hard_moved

    def _masked_mean(tensor, mask):
        mask = mask.to(dtype=torch.bool, device=tensor.device)
        if mask.any():
            return tensor[mask].mean()
        return tensor.sum() * 0.0

    moved_bce = _masked_mean(per_point_bce, moved_valid)
    static_bce = _masked_mean(per_point_bce, static_valid)
    move_prediction_loss = 0.5 * (moved_bce + static_bce)

    with torch.no_grad():
        probs = torch.sigmoid(move_logits)
        pred_moved = probs >= 0.5
        tp = (pred_moved & moved_valid).sum().float()
        fp = (pred_moved & static_valid).sum().float()
        fn = ((~pred_moved) & moved_valid).sum().float()
        tn = ((~pred_moved) & static_valid).sum().float()
        valid_count = valid.sum().float().clamp(min=1.0)
        hard_acc = ((pred_moved == hard_moved) & valid).sum().float() / valid_count
        precision = tp / (tp + fp).clamp(min=1.0)
        recall = tp / (tp + fn).clamp(min=1.0)
        static_recall = tn / (tn + fp).clamp(min=1.0)

    metrics = {
        "move_prediction_loss": move_prediction_loss,
        "move_prediction_bce_moved": moved_bce,
        "move_prediction_bce_static": static_bce,
        "move_prediction_target_mean": move_targets[valid].mean() if valid.any() else move_targets.sum() * 0.0,
        "move_prediction_prob_mean": probs[valid].mean() if valid.any() else probs.sum() * 0.0,
        "move_prediction_accuracy": hard_acc,
        "move_prediction_precision": precision,
        "move_prediction_recall": recall,
        "move_prediction_static_recall": static_recall,
    }
    return move_prediction_loss, metrics


@torch.no_grad()
def compute_single_output_metrics(
    args,
    output_flows,
    gt_flows,
    per_point_loss,
    weights,
    moved,
    static,
    pred_exists_supervised,
    log_var,
    pred_norm,
    var_floor: float,
    var_ceiling: float,
):
    l2 = torch.norm(output_flows - gt_flows, dim=-1)  # (B,T,NS)

    metrics = metrics_utils.collect_metrics(
        args,
        per_point_loss,
        weights,
        moved,
        static,
        pred_exists_supervised,
        l2,
        log_var,
        pred_norm,
        var_floor=var_floor,
        var_ceiling=var_ceiling,
    )

    return metrics


def internal_loss_fn(
    model,
    outputs,
    data_dict,
    training: bool,
    var_floor: float,
    var_ceiling: float,
    sim_domain_keywords: list[str],
    sim_var_const: float,
    uncertainty_logvar_weight: float,
):
    # ------------------------------------------------------------------ #
    #  Unpack tensors & masks                                            #
    # ------------------------------------------------------------------ #
    output_scene_flows = outputs["scene_flows"]  # (B,T,NS,3)
    gt_scene_flows = data_dict["gt_scene_flows"]  # (B,T,NS,3)
    weights = data_dict["point_weights"].to(model.device)
    assert torch.all(weights >= 0), "point_weights must be non-negative"
    weights = weights.clamp_min(0)

    gt_target = data_dict["gt_scene_flows_relative"]
    output_norm = outputs["scene_relative_norm"]
    gt_target_norm = model.normalize(gt_target)

    log_var = outputs["log_var"]

    moved = data_dict["scene_moved_mask"].squeeze(-1).bool()
    static = data_dict["scene_static_mask"].squeeze(-1).bool()
    context = data_dict["scene_context_mask"].squeeze(-1).bool()
    exists = data_dict["scene_exists"].squeeze(-1).bool()
    supervised = data_dict["scene_supervised_mask"].squeeze(-1).bool()
    pred = ~context
    pred_exists_supervised = pred & exists & supervised

    # ------------------------------------------------------------------ #
    #  Compute loss and metrics                                          #
    # ------------------------------------------------------------------ #
    def _prepare_log_var(_log_var):
        Bv, Tv, Nsv = _log_var.shape[:3]
        sim_b = torch.tensor(
            [any(k in dom for k in sim_domain_keywords) for dom in data_dict["__domain__"]],
            device=model.device,
            dtype=torch.bool,
        )
        if sim_b.any():
            sim_mask = sim_b.view(Bv, 1, 1, 1).expand_as(_log_var)
            real_b = ~sim_b
            if real_b.any():
                s_min_tmp = math.log(var_floor)
                s_max_tmp = math.log(var_ceiling)
                clamped = _log_var.detach().clamp(min=s_min_tmp, max=s_max_tmp)
                real_logvar = clamped[real_b]
                real_var_mean = real_logvar.exp().mean().clamp(
                    min=float(var_floor), max=float(var_ceiling)
                )
                const_val = float(real_var_mean.log().item())
            else:
                const_val = math.log(sim_var_const)
            const_tensor = torch.full_like(_log_var, fill_value=const_val)
            _log_var = torch.where(sim_mask, const_tensor, _log_var)
        s_min = math.log(var_floor)
        s_max = math.log(var_ceiling)
        return _log_var.clamp(min=s_min, max=s_max)

    log_var = _prepare_log_var(log_var)

    log_dict = {}
    total_loss, per_point_loss, loss_terms = compute_single_output_loss(
        model.args,
        output_norm,
        gt_target_norm,
        weights,
        pred_exists_supervised,
        log_var,
        var_floor=var_floor,
        var_ceiling=var_ceiling,
        uncertainty_logvar_weight=uncertainty_logvar_weight,
        moved_mask=moved,
        static_mask=static,
        pred_relative=outputs["scene_relative"],
        gt_relative=gt_target,
        selector_gt=data_dict["scene_selector_gt"],
    )
    if bool(getattr(model.args, "use_move_prediction_head", False)):
        move_prediction_loss, move_metrics = compute_move_prediction_loss(
            model.args,
            outputs,
            gt_target,
            pred_exists_supervised,
        )
        move_weight = float(getattr(model.args, "move_prediction_loss_weight", 0.1))
        total_loss = total_loss + move_weight * move_prediction_loss
        for key, value in move_metrics.items():
            loss_terms[key] = value
        loss_terms["move_prediction_weighted_loss"] = move_weight * move_prediction_loss

    metrics = compute_single_output_metrics(
        model.args,
        output_scene_flows,
        gt_scene_flows,
        per_point_loss,
        weights,
        moved,
        static,
        pred_exists_supervised,
        log_var,
        outputs["pred"],
        var_floor=var_floor,
        var_ceiling=var_ceiling,
    )
    metric_keys = list(metrics.keys())
    log_dict.update(metrics)
    for key, value in loss_terms.items():
        log_dict[key] = value.detach().cpu().item()
    log_dict["total_loss"] = total_loss.detach().cpu().item()

    log_dict = metrics_utils.collect_per_domain_metrics(
        model.args,
        model.device,
        data_dict,
        output_scene_flows,
        gt_scene_flows,
        per_point_loss,
        weights,
        moved,
        static,
        pred_exists_supervised,
        log_var,
        metric_keys,
        log_dict,
        outputs["pred"],
        lambda *args: compute_single_output_metrics(
            model.args,
            *args,
            var_floor=var_floor,
            var_ceiling=var_ceiling,
        ),
    )

    # ------------------------------------------------------------------ #
    #  Filtered metrics (eval only)                                      #
    # ------------------------------------------------------------------ #
    if (not training) and ("scene_filter_mask" in data_dict):
        conf_seed = data_dict.get("confidence_seed", None)
        if conf_seed is not None:
            if isinstance(conf_seed, torch.Tensor):
                if conf_seed.numel() == 1:
                    seed_val = int(conf_seed.item())
                else:
                    unique = torch.unique(conf_seed.detach().view(-1))
                    assert unique.numel() == 1, "Mixed confidence seeds detected in batch"
                    seed_val = int(unique.item())
            else:
                seed_val = int(conf_seed)
            assert seed_val == int(model.args.seed), (
                f"Confidence seed mismatch: file seed={seed_val}, args.seed={model.args.seed}"
            )

        l2_full = torch.norm(output_scene_flows - gt_scene_flows, dim=-1)  # (B,T,Ns)

        filt = data_dict["scene_filter_mask"].to(l2_full.device).bool()
        valid = filt & pred_exists_supervised

        def _masked_mean(x, m):
            if m.any():
                return x[m].mean().item()
            return float("nan")

        log_dict["filtered_l2/mean"] = _masked_mean(l2_full, valid)
        log_dict["filtered_l2_moved/mean"] = _masked_mean(l2_full, valid & moved)
        log_dict["filtered_l2_static/mean"] = _masked_mean(l2_full, valid & static)

    return total_loss, log_dict


def loss_fn(
    model,
    outputs,
    data_dict,
    training: bool,
    var_floor: float,
    var_ceiling: float,
    sim_domain_keywords: list[str],
    sim_var_const: float,
    uncertainty_logvar_weight: float,
):
    def _compute_loss():
        return internal_loss_fn(
            model,
            outputs,
            data_dict,
            training,
            var_floor=var_floor,
            var_ceiling=var_ceiling,
            sim_domain_keywords=sim_domain_keywords,
            sim_var_const=sim_var_const,
            uncertainty_logvar_weight=uncertainty_logvar_weight,
        )

    return safe_loss_computation(
        _compute_loss,
        f"loss computation (training={training})",
    )
