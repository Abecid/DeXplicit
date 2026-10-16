from __future__ import annotations

import torch

from training.long_horizon_common import (
    _gather_btnh,
    _horizon_mask,
    _masked_sum_count,
)
from training.long_horizon_model_core import LongHorizonModelCoreMixin
from training.long_horizon_model_physics import LongHorizonPhysicsLossMixin
from training.long_horizon_model_structure import (
    LongHorizonStructureLossMixin,
)
from training.object_jepa_hidden_only import (
    HiddenOnlyObjectStateJEPAV2ExperimentalBaseModel,
)


class LongHorizonAblationModel(
    LongHorizonModelCoreMixin,
    LongHorizonStructureLossMixin,
    LongHorizonPhysicsLossMixin,
    HiddenOnlyObjectStateJEPAV2ExperimentalBaseModel,
):
    def loss_fn(self, outputs, data, training=True):
        total, log = super().loss_fn(outputs, data, training=training)
        log["long_horizon/stage"] = float(self.long_horizon_stage)

        if "gt_relative_edge_loss" in self.long_horizon_features:
            edge_loss, _metrics = self._edge_terms(outputs, data)
            total = total + self.long_horizon_edge_weight * edge_loss
            log["long_horizon/edge_loss"] = float(
                edge_loss.detach().cpu()
            )
            log["long_horizon/edge_weighted"] = float(
                (
                    self.long_horizon_edge_weight * edge_loss
                ).detach().cpu()
            )

        if "per_hand_contact_distance" in self.long_horizon_features:
            contact_loss = self._contact_aux_terms(outputs, data)
            total = total + self.long_horizon_contact_weight * contact_loss
            log["long_horizon/contact_aux"] = float(
                contact_loss.detach().cpu()
            )
            log["long_horizon/contact_aux_weighted"] = float(
                (
                    self.long_horizon_contact_weight * contact_loss
                ).detach().cpu()
            )

        if "particle_deformation_bottleneck" in self.long_horizon_features:
            balance, entropy, compact = self._particle_regularization(
                outputs, data
            )
            particle_weighted = (
                self.long_horizon_particle_balance_weight * balance
                + self.long_horizon_particle_entropy_weight * entropy
                + self.long_horizon_particle_compact_weight * compact
            )
            total = total + particle_weighted
            log.update(
                {
                    "long_horizon/particle_balance": float(
                        balance.detach().cpu()
                    ),
                    "long_horizon/particle_entropy": float(
                        entropy.detach().cpu()
                    ),
                    "long_horizon/particle_compact": float(
                        compact.detach().cpu()
                    ),
                    "long_horizon/particle_weighted": float(
                        particle_weighted.detach().cpu()
                    ),
                }
            )

        if "physical_probe" in self.long_horizon_features:
            physical = self._physical_probe_loss(outputs, data)
            total = total + self.long_horizon_physical_weight * physical
            log["long_horizon/physical_probe"] = float(
                physical.detach().cpu()
            )
            log["long_horizon/physical_probe_weighted"] = float(
                (
                    self.long_horizon_physical_weight * physical
                ).detach().cpu()
            )

        if (
            "contact_consistency" in self.long_horizon_features
            or "stick_contact" in self.long_horizon_features
        ):
            consistency, stick = self._contact_physics_losses(outputs, data)
            weighted = (
                self.long_horizon_consistency_weight * consistency
                + self.long_horizon_stick_weight * stick
            )
            total = total + weighted
            log.update(
                {
                    "long_horizon/contact_consistency": float(
                        consistency.detach().cpu()
                    ),
                    "long_horizon/stick": float(stick.detach().cpu()),
                    "long_horizon/contact_physics_weighted": float(
                        weighted.detach().cpu()
                    ),
                }
            )

        log["total_loss"] = float(total.detach().cpu())
        return total, log

    @torch.no_grad()
    def long_horizon_eval_sums(self, batch, outputs):
        """Count-weighted structure/contact metrics shared by every stage."""
        result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        _edge_loss, edge_metrics = self._edge_terms(outputs, batch)
        for name, pair in edge_metrics.items():
            result[f"long_horizon/structure/{name}"] = pair

        (
            target,
            target_valid,
            _nearest,
            indices,
            selected_valid,
        ) = self._contact_targets(
            outputs, batch, use_predicted_geometry=False
        )
        (
            geometry,
            geometry_valid,
            _unused,
            _indices,
            _selected,
        ) = self._contact_targets(
            outputs, batch, use_predicted_geometry=True
        )
        valid = target_valid & geometry_valid & selected_valid[
            :, None, :, :, None
        ]
        error = (geometry - target).abs()
        ranges = (
            ("geometry_contact_mae_t01_t14", 1, 14),
            (
                "geometry_contact_mae_t15_t34",
                15,
                geometry.shape[1] - 1,
            ),
            (
                "geometry_contact_mae_t01_t34",
                1,
                geometry.shape[1] - 1,
            ),
            (
                "geometry_contact_mae_t34",
                geometry.shape[1] - 1,
                geometry.shape[1] - 1,
            ),
        )
        for name, start, end in ranges:
            horizon = _horizon_mask(
                geometry.shape[1], start, end, geometry.device
            )[None, :, None, None, None]
            result[f"long_horizon/contact/{name}"] = _masked_sum_count(
                error, valid & horizon
            )

        threshold = self.long_horizon_contact_threshold
        gt_contact = target < threshold
        pred_contact = geometry < threshold
        future = _horizon_mask(
            geometry.shape[1],
            1,
            geometry.shape[1] - 1,
            geometry.device,
        )[None, :, None, None, None]
        metric_valid = valid & future
        true_positive = (
            gt_contact & pred_contact & metric_valid
        ).sum()
        gt_positive = (gt_contact & metric_valid).sum()
        pred_positive = (pred_contact & metric_valid).sum()
        result["long_horizon/contact/recall_5mm"] = (
            true_positive.to(error.dtype),
            gt_positive,
        )
        result["long_horizon/contact/precision_5mm"] = (
            true_positive.to(error.dtype),
            pred_positive,
        )

        if "per_hand_contact_distance" in self.long_horizon_features:
            predicted_contact = _gather_btnh(
                outputs["long_horizon_contact_distance"], indices
            )
            contact_head_error = (predicted_contact - target).abs()
            result["long_horizon/contact/head_mae_t01_t34"] = (
                _masked_sum_count(
                    contact_head_error, valid & future
                )
            )
        return result
