"""Low-rank object-local deformation through persistent latent particles.

Research provenance
-------------------
This stage is inspired by Latent Particle World Models (LPWM,
arXiv:2603.04553) and its official implementation:
https://github.com/taldatech/lpwm

It is NOT a port of DLPv3/LPWM.  LPWM discovers 2-D appearance particles from
video, learns stochastic per-particle latent actions with posterior/prior KL,
uses AdaLN-conditioned causal Transformer dynamics, and autoregressively decodes
future frames.  PointWorld already has supervised object IDs, canonical 3-D
points, persistent correspondences, and a planned hand trajectory.  We therefore
borrow only the structural hypothesis:

    many dense points -> a small set of persistent local particles/parts
                    -> coherent per-particle dynamics -> dense deformation.

The deterministic soft assignments, canonical 3-D motion decoder, balance /
entropy / compactness regularizers, and composition under the existing object
SE(3) are PointWorld-specific.  In particular, ``K=16`` and temperature ``0.5``
are pilot values, not LPWM defaults.  LPWM's released Sketchy-action config uses
64 prior particles, 45 encoded particles, hidden width 512, six dynamics layers,
eight heads, ``beta_dyn=0.2``, and ``beta_dyn_rec=1.0``.  A serious stage-6
study should sweep at least K in {16, 32, 64} instead of treating 16 as proven.

See ``training/long_horizon_provenance.py`` for the complete audit ledger.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from training.long_horizon_common import contact_strength


class LongHorizonParticleMixin:
    def _particle_residual(
        self,
        point_features: torch.Tensor,
        encoded: torch.Tensor,
        contact_distance: torch.Tensor,
        hand_token: torch.Tensor,
        ids: torch.Tensor,
        valid: torch.Tensor,
        object_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, points, _ = point_features.shape
        particle_count = int(self.long_horizon_particle_count)
        residual = point_features.new_zeros(batch, self.T, points, 3)
        assignment_out = point_features.new_zeros(
            batch, points, particle_count
        )
        motion_out = point_features.new_zeros(
            batch,
            object_valid.shape[1],
            self.T,
            particle_count,
            3,
        )
        strength = contact_strength(
            contact_distance, self.long_horizon_contact_tau
        )
        for batch_index in range(batch):
            for object_index in range(object_valid.shape[1]):
                if not bool(object_valid[batch_index, object_index]):
                    continue
                mask = valid[batch_index].bool() & (
                    ids[batch_index] == object_index
                )
                if mask.sum() < 3:
                    continue
                features = point_features[batch_index, mask]

                # LPWM-inspired structural bottleneck, PointWorld-specific
                # parameterization: a persistent soft assignment from supervised
                # 3-D object points to K latent parts. LPWM instead discovers 2-D
                # DLP particles and models full particle attributes.
                logits = self.long_horizon_particle_assignment(
                    features.float()
                )
                assignment = F.softmax(
                    logits
                    / max(self.long_horizon_particle_temperature, 1e-3),
                    dim=-1,
                ).to(point_features.dtype)
                assignment_out[batch_index, mask] = assignment
                usage = assignment.sum(dim=0).clamp_min(1e-6)
                particle_feature = torch.einsum(
                    "nk,nc->kc", assignment.float(), features.float()
                ) / usage.float().unsqueeze(-1)

                if self._has("per_hand_contact_distance"):
                    # LPWM motivates per-particle transition context.  Our
                    # contact-weighted known-hand context replaces LPWM's
                    # stochastic inverse-model/policy-prior latent action; no KL
                    # or autoregressive latent-action model is implemented here.
                    point_strength = strength[batch_index, 1:, mask]
                    particle_strength = torch.einsum(
                        "nk,tnh->tkh",
                        assignment.float(),
                        point_strength.float(),
                    ) / usage.float().view(1, -1, 1)
                    contact_context = torch.einsum(
                        "tkh,thc->tkc",
                        particle_strength,
                        self.long_horizon_particle_contact(
                            hand_token[batch_index, 1:].float()
                        ),
                    )
                else:
                    contact_context = particle_feature.new_zeros(
                        self.T - 1,
                        particle_count,
                        self.dim,
                    )

                particle_time = (
                    particle_feature[:, None]
                    + encoded[batch_index, object_index][None]
                    + contact_context.permute(1, 0, 2)
                )
                particle_motion = torch.tanh(
                    self.long_horizon_particle_motion(particle_time)
                ) * self.long_horizon_max_deformation
                decoded = torch.einsum(
                    "nk,ktc->tnc",
                    assignment.float(),
                    particle_motion.float(),
                ).to(point_features.dtype)
                decoded -= decoded.mean(dim=1, keepdim=True)
                residual[batch_index, 1:, mask] = decoded
                motion_out[batch_index, object_index, 1:] = (
                    particle_motion.permute(1, 0, 2)
                )
        return residual, assignment_out, motion_out

    def _keep_new_parameters_live(
        self, reference: torch.Tensor
    ) -> torch.Tensor:
        live = reference.sum() * 0.0
        for name, parameter in self.named_parameters():
            if name.startswith("long_horizon_") or name.startswith("score_se3_"):
                live = live + parameter.sum() * 0.0
        return live
