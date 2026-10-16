"""Contact summaries for the score-aligned residual-SE(3) campaign.

The residual pose adapter reserves ten contact channels at every object/time
token.  Each hand contributes five values in this exact order::

    [canonical_anchor_x, canonical_anchor_y, canonical_anchor_z,
     contact_mass, log1p(min_distance / tau)]

``metric_pose`` fills all ten channels with literal zeros.  ``predicted_contact`` summarizes the
model's predicted continuous point-to-left/right-hand distances.  ``oracle_contact`` uses
the identical summarizer on distances computed from future ground-truth
geometry, but that path is deliberately exposed through a separate function
and requires an explicit oracle-access flag.

No entry point accepts a generic training batch.  In particular, the predicted_contact API
has no future-geometry argument, which makes accidentally routing
``gt_scene_flows`` into a deployable model a call-site error rather than a
convention reviewers must remember.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from training.long_horizon_common import (
    _split_hand_points,
    batched_min_hand_distance,
)
from training.residual_se3_adapter import (
    CONTACT_CONDITIONING_DIM as ADAPTER_CONTACT_CONDITIONING_DIM,
)


CONTACT_HAND_COUNT = 2
CONTACT_FEATURES_PER_HAND = 5
CONTACT_CONDITIONING_DIM = ADAPTER_CONTACT_CONDITIONING_DIM
if CONTACT_HAND_COUNT * CONTACT_FEATURES_PER_HAND != CONTACT_CONDITIONING_DIM:
    raise RuntimeError(
        "contact feature layout disagrees with residual adapter input width"
    )
CONTACT_FEATURE_NAMES = (
    "left_anchor_x",
    "left_anchor_y",
    "left_anchor_z",
    "left_mass",
    "left_log_min_distance",
    "right_anchor_x",
    "right_anchor_y",
    "right_anchor_z",
    "right_mass",
    "right_log_min_distance",
)


@dataclass(frozen=True)
class ContactConditioningResult:
    """Fixed-width contact features and masks.

    Attributes:
        features: ``(B,O,T,10)`` contact conditioning tensor.
        valid: ``(B,O,T,2)`` mask indicating that the object/hand distance is
            geometrically defined (at least one valid object point and one
            valid hand point).
        anchor_valid: ``(B,O,T,2)`` mask indicating non-negligible Gaussian
            contact mass.  A geometrically valid but distant hand can therefore
            have ``valid=True`` and ``anchor_valid=False``; its log minimum
            distance remains informative while its anchor and mass are zero.
    """

    features: torch.Tensor
    valid: torch.Tensor
    anchor_valid: torch.Tensor

    def detached(self) -> "ContactConditioningResult":
        """Stop all pose-loss gradients through the contact summary."""

        return ContactConditioningResult(
            features=self.features.detach(),
            valid=self.valid.detach(),
            anchor_valid=self.anchor_valid.detach(),
        )


def _require_floating(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {value.dtype}")


def _normalize_point_valid(
    point_valid: torch.Tensor,
    *,
    batch: int,
    timesteps: int,
    points: int,
    device: torch.device,
) -> torch.Tensor:
    value = point_valid.to(device=device)
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.shape == (batch, points):
        value = value[:, None].expand(-1, timesteps, -1)
    elif value.shape != (batch, timesteps, points):
        raise ValueError(
            "point_valid must be (B,N), (B,T,N), or (B,T,N,1); "
            f"got {tuple(point_valid.shape)} for B={batch}, T={timesteps}, N={points}"
        )
    return value.bool()


def _normalize_object_valid(
    object_valid: torch.Tensor,
    *,
    batch: int,
    timesteps: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    value = object_valid.to(device=device)
    if value.ndim == 2 and value.shape[0] == batch:
        objects = int(value.shape[1])
        value = value[:, :, None].expand(-1, -1, timesteps)
    elif value.ndim == 3 and value.shape[0] == batch and value.shape[2] == timesteps:
        objects = int(value.shape[1])
    else:
        raise ValueError(
            "object_valid must be (B,O) or (B,O,T); "
            f"got {tuple(object_valid.shape)} for B={batch}, T={timesteps}"
        )
    return value.bool(), objects


def hand_side_validity(hand_valid: torch.Tensor) -> torch.Tensor:
    """Reduce raw HOT3D hand-point validity to left/right validity.

    HOT3D hand points follow the repository convention used by
    :func:`training.long_horizon_common._split_hand_points`: the first floor
    half is left and the remaining points are right.
    """

    if not isinstance(hand_valid, torch.Tensor):
        raise TypeError("hand_valid must be a torch.Tensor")
    if hand_valid.ndim == 4 and hand_valid.shape[-1] == 1:
        hand_valid = hand_valid.squeeze(-1)
    if hand_valid.ndim != 3:
        raise ValueError(
            f"hand_valid must be (B,T,R) or (B,T,R,1), got {tuple(hand_valid.shape)}"
        )
    hand_points = int(hand_valid.shape[2])
    if hand_points < 2:
        raise ValueError("at least two hand points are required to form left/right groups")
    left_count = hand_points // 2
    valid = hand_valid.bool()
    return torch.stack(
        (
            valid[:, :, :left_count].any(dim=2),
            valid[:, :, left_count:].any(dim=2),
        ),
        dim=-1,
    )


def zero_contact_conditioning(
    object_valid: torch.Tensor,
    timesteps: int,
    *,
    reference: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> ContactConditioningResult:
    """Return metric_pose's literal-zero, capacity-matched contact channels."""

    if not isinstance(object_valid, torch.Tensor):
        raise TypeError("object_valid must be a torch.Tensor")
    if int(timesteps) < 0:
        raise ValueError("timesteps must be non-negative")
    if reference is not None:
        _require_floating("reference", reference)
        if dtype is not None and dtype != reference.dtype:
            raise ValueError("dtype disagrees with reference.dtype")
        if device is not None and torch.device(device) != reference.device:
            raise ValueError("device disagrees with reference.device")
        dtype = reference.dtype
        device = reference.device
    dtype = dtype or torch.float32
    device = torch.device(device) if device is not None else object_valid.device
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("contact conditioning dtype must be floating point")

    if object_valid.ndim not in (2, 3):
        raise ValueError("object_valid must be (B,O) or (B,O,T)")
    batch, objects = int(object_valid.shape[0]), int(object_valid.shape[1])
    if object_valid.ndim == 3 and int(object_valid.shape[2]) != int(timesteps):
        raise ValueError("temporal object_valid length does not match timesteps")
    features = torch.zeros(
        batch,
        objects,
        int(timesteps),
        CONTACT_CONDITIONING_DIM,
        dtype=dtype,
        device=device,
    )
    mask = torch.zeros(
        batch,
        objects,
        int(timesteps),
        CONTACT_HAND_COUNT,
        dtype=torch.bool,
        device=device,
    )
    return ContactConditioningResult(features, mask, mask.clone())


def _summarize_distance_field(
    *,
    canonical_xyz: torch.Tensor,
    point_object_id: torch.Tensor,
    point_valid: torch.Tensor,
    object_valid: torch.Tensor,
    point_hand_distance: torch.Tensor,
    distance_valid: torch.Tensor,
    tau: float,
    anchor_mass_epsilon: float,
) -> ContactConditioningResult:
    """Shared, differentiable predicted_contact/oracle_contact distance-field summarizer."""

    _require_floating("canonical_xyz", canonical_xyz)
    _require_floating("point_hand_distance", point_hand_distance)
    if point_hand_distance.ndim != 4 or point_hand_distance.shape[-1] != 2:
        raise ValueError(
            "point_hand_distance must be (B,T,N,2), got "
            f"{tuple(point_hand_distance.shape)}"
        )
    batch, timesteps, points, hands = point_hand_distance.shape
    if hands != CONTACT_HAND_COUNT:
        raise ValueError("point_hand_distance must contain exactly left/right channels")
    if canonical_xyz.shape != (batch, points, 3):
        raise ValueError(
            f"canonical_xyz must be {(batch, points, 3)}, got {tuple(canonical_xyz.shape)}"
        )
    if point_object_id.shape != (batch, points):
        raise ValueError(
            f"point_object_id must be {(batch, points)}, got {tuple(point_object_id.shape)}"
        )
    if distance_valid.shape != point_hand_distance.shape:
        raise ValueError("distance_valid must match point_hand_distance")
    if not float(tau) > 0.0:
        raise ValueError("tau must be positive")
    if not float(anchor_mass_epsilon) >= 0.0:
        raise ValueError("anchor_mass_epsilon must be non-negative")

    device = point_hand_distance.device
    output_dtype = point_hand_distance.dtype
    canonical = canonical_xyz.to(device=device, dtype=torch.float32)
    ids = point_object_id.to(device=device).long()
    valid_points = _normalize_point_valid(
        point_valid,
        batch=batch,
        timesteps=timesteps,
        points=points,
        device=device,
    )
    temporal_object_valid, objects = _normalize_object_valid(
        object_valid,
        batch=batch,
        timesteps=timesteps,
        device=device,
    )
    valid_distance = distance_valid.to(device=device).bool()
    valid_distance = valid_distance & valid_points.unsqueeze(-1)

    relevant_canonical = valid_points.any(dim=1) & (ids >= 0) & (ids < objects)
    finite_canonical = torch.isfinite(canonical).all(dim=-1)
    if bool((relevant_canonical & ~finite_canonical).any()):
        raise ValueError("canonical_xyz contains non-finite coordinates for a valid object point")
    canonical = torch.where(
        finite_canonical.unsqueeze(-1), canonical, torch.zeros_like(canonical)
    )

    distance = point_hand_distance.float()
    finite_distance = torch.isfinite(distance)
    if bool((valid_distance & ~finite_distance).any()):
        raise ValueError("point_hand_distance contains a non-finite valid entry")
    if bool((valid_distance & (distance < 0)).any()):
        raise ValueError("point_hand_distance contains a negative valid entry")
    safe_distance = torch.where(valid_distance, distance, torch.zeros_like(distance))

    features = distance.new_zeros(
        batch, objects, timesteps, CONTACT_CONDITIONING_DIM
    )
    summary_valid = torch.zeros(
        batch,
        objects,
        timesteps,
        CONTACT_HAND_COUNT,
        dtype=torch.bool,
        device=device,
    )
    anchor_valid_output = torch.zeros_like(summary_valid)
    tau_value = float(tau)

    if points == 0:
        return ContactConditioningResult(
            features=features.to(output_dtype),
            valid=summary_valid,
            anchor_valid=anchor_valid_output,
        )

    # Loop over objects rather than materializing a (B,T,O,N,2) membership
    # tensor.  HOT3D has many points and few objects; this keeps peak memory
    # bounded while all reductions over points remain vectorized.
    for object_index in range(objects):
        membership = ids == object_index
        object_point_valid = valid_distance & membership[:, None, :, None]
        available = object_point_valid.any(dim=2)
        available = available & temporal_object_valid[:, object_index, :, None]

        strength = torch.exp(
            -0.5 * (safe_distance / tau_value).square()
        ) * object_point_valid.to(distance.dtype)
        mass = strength.sum(dim=2)
        has_anchor = available & (mass > float(anchor_mass_epsilon))

        numerator = torch.einsum("btnh,bnc->bthc", strength, canonical)
        anchor = numerator / mass.clamp_min(
            max(float(anchor_mass_epsilon), torch.finfo(distance.dtype).tiny)
        ).unsqueeze(-1)
        anchor = torch.where(
            has_anchor.unsqueeze(-1), anchor, torch.zeros_like(anchor)
        )

        infinity = torch.full_like(safe_distance, torch.inf)
        masked_distance = torch.where(object_point_valid, safe_distance, infinity)
        minimum = masked_distance.min(dim=2).values
        minimum = torch.where(available, minimum, torch.zeros_like(minimum))
        log_minimum = torch.log1p(minimum / tau_value)
        mass = torch.where(has_anchor, mass, torch.zeros_like(mass))

        per_hand = torch.cat(
            (anchor, mass.unsqueeze(-1), log_minimum.unsqueeze(-1)), dim=-1
        )
        per_hand = torch.where(
            available.unsqueeze(-1), per_hand, torch.zeros_like(per_hand)
        )
        features[:, object_index] = per_hand.reshape(
            batch, timesteps, CONTACT_CONDITIONING_DIM
        )
        summary_valid[:, object_index] = available
        anchor_valid_output[:, object_index] = has_anchor

    return ContactConditioningResult(
        features=features.to(output_dtype),
        valid=summary_valid,
        anchor_valid=anchor_valid_output,
    )


def predicted_contact_conditioning(
    *,
    canonical_xyz: torch.Tensor,
    point_object_id: torch.Tensor,
    point_valid: torch.Tensor,
    object_valid: torch.Tensor,
    predicted_point_hand_distance: torch.Tensor,
    hand_valid: torch.Tensor,
    tau: float = 0.02,
    anchor_mass_epsilon: float = 1e-8,
    detach: bool = True,
) -> ContactConditioningResult:
    """Build predicted_contact features exclusively from deployable inference inputs.

    ``predicted_point_hand_distance`` follows the existing contact-head layout
    ``(B,T,N,2)``.  The default stop-gradient behavior is the campaign's first
    causal experiment: the contact head learns from its auxiliary target while
    the pose loss cannot game the distance field.

    This function intentionally has no future-object-geometry parameter.
    """

    if predicted_point_hand_distance.ndim != 4:
        raise ValueError("predicted_point_hand_distance must be (B,T,N,2)")
    batch, timesteps = predicted_point_hand_distance.shape[:2]
    side_valid = hand_side_validity(hand_valid).to(
        device=predicted_point_hand_distance.device
    )
    if side_valid.shape != (batch, timesteps, CONTACT_HAND_COUNT):
        raise ValueError(
            "hand_valid and predicted_point_hand_distance must share B,T"
        )
    distance_valid = side_valid[:, :, None, :].expand_as(
        predicted_point_hand_distance
    )
    result = _summarize_distance_field(
        canonical_xyz=canonical_xyz,
        point_object_id=point_object_id,
        point_valid=point_valid,
        object_valid=object_valid,
        point_hand_distance=predicted_point_hand_distance,
        distance_valid=distance_valid,
        tau=tau,
        anchor_mass_epsilon=anchor_mass_epsilon,
    )
    return result.detached() if bool(detach) else result


@torch.no_grad()
def oracle_contact_conditioning(
    *,
    canonical_xyz: torch.Tensor,
    point_object_id: torch.Tensor,
    object_valid: torch.Tensor,
    future_object_xyz: torch.Tensor,
    future_point_valid: torch.Tensor,
    hand_xyz: torch.Tensor,
    hand_valid: torch.Tensor,
    allow_future_ground_truth: bool = False,
    tau: float = 0.02,
    anchor_mass_epsilon: float = 1e-8,
    distance_chunk_bt: int = 32,
) -> ContactConditioningResult:
    """Build the non-deployable oracle_contact oracle from future ground-truth geometry.

    Access is fail-closed: callers must pass the literal boolean
    ``allow_future_ground_truth=True``.  metric_pose/predicted_contact never call this function and
    their public API cannot accept ``future_object_xyz``.
    """

    if allow_future_ground_truth is not True:
        raise PermissionError(
            "oracle_contact oracle contact requires explicit allow_future_ground_truth=True"
        )
    _require_floating("future_object_xyz", future_object_xyz)
    _require_floating("hand_xyz", hand_xyz)
    if int(distance_chunk_bt) < 1:
        raise ValueError("distance_chunk_bt must be positive")
    if future_object_xyz.ndim != 4 or future_object_xyz.shape[-1] != 3:
        raise ValueError("future_object_xyz must be (B,T,N,3)")
    if hand_xyz.ndim != 4 or hand_xyz.shape[-1] != 3:
        raise ValueError("hand_xyz must be (B,T,R,3)")
    batch, timesteps, points, _ = future_object_xyz.shape
    if hand_xyz.shape[:2] != (batch, timesteps):
        raise ValueError("future_object_xyz and hand_xyz must share B,T")
    point_valid = _normalize_point_valid(
        future_point_valid,
        batch=batch,
        timesteps=timesteps,
        points=points,
        device=future_object_xyz.device,
    )
    raw_hand_valid = hand_valid.to(device=future_object_xyz.device)
    if raw_hand_valid.ndim == 4 and raw_hand_valid.shape[-1] == 1:
        raw_hand_valid = raw_hand_valid.squeeze(-1)
    if raw_hand_valid.shape != hand_xyz.shape[:-1]:
        raise ValueError("hand_valid must match hand_xyz without XYZ")
    raw_hand_valid = raw_hand_valid.bool()

    object_points = future_object_xyz.float()
    finite_points = torch.isfinite(object_points).all(dim=-1)
    if bool((point_valid & ~finite_points).any()):
        raise ValueError("future_object_xyz contains non-finite valid coordinates")
    object_points = torch.where(
        finite_points.unsqueeze(-1), object_points, torch.zeros_like(object_points)
    )
    hands = hand_xyz.to(device=object_points.device, dtype=torch.float32)
    finite_hands = torch.isfinite(hands).all(dim=-1)
    if bool((raw_hand_valid & ~finite_hands).any()):
        raise ValueError("hand_xyz contains non-finite valid coordinates")
    hands = torch.where(
        finite_hands.unsqueeze(-1), hands, torch.zeros_like(hands)
    )

    distances: list[torch.Tensor] = []
    valid_sides: list[torch.Tensor] = []
    for side_points, side_valid in _split_hand_points(hands, raw_hand_valid):
        side_distance, side_distance_valid, _ = batched_min_hand_distance(
            object_points,
            point_valid,
            side_points,
            side_valid,
            chunk_bt=int(distance_chunk_bt),
            return_indices=False,
        )
        distances.append(side_distance)
        valid_sides.append(side_distance_valid)
    point_hand_distance = torch.stack(distances, dim=-1)
    distance_valid = torch.stack(valid_sides, dim=-1)
    result = _summarize_distance_field(
        canonical_xyz=canonical_xyz,
        point_object_id=point_object_id,
        point_valid=point_valid,
        object_valid=object_valid,
        point_hand_distance=point_hand_distance,
        distance_valid=distance_valid,
        tau=tau,
        anchor_mass_epsilon=anchor_mass_epsilon,
    )
    return ContactConditioningResult(
        features=result.features.to(future_object_xyz.dtype),
        valid=result.valid,
        anchor_valid=result.anchor_valid,
    )


class ContactSE3Conditioner(nn.Module):
    """Mode-locked façade used by the metric_pose/predicted_contact/oracle_contact model integration.

    The mode-specific methods have deliberately disjoint signatures.  A
    deployable predicted_contact conditioner therefore cannot be called with oracle geometry,
    and an oracle_contact conditioner cannot be constructed without an explicit access
    grant.
    """

    def __init__(
        self,
        candidate: str,
        *,
        tau: float = 0.02,
        anchor_mass_epsilon: float = 1e-8,
        detach_predicted: bool = True,
        allow_future_ground_truth: bool = False,
    ) -> None:
        super().__init__()
        if candidate not in {"metric_pose", "predicted_contact", "oracle_contact"}:
            raise ValueError("contact conditioner candidate must be metric_pose, predicted_contact, or oracle_contact")
        if candidate == "oracle_contact" and allow_future_ground_truth is not True:
            raise PermissionError(
                "constructing oracle_contact requires allow_future_ground_truth=True"
            )
        if candidate != "oracle_contact" and allow_future_ground_truth:
            raise ValueError(
                "future-ground-truth access is forbidden for deployable metric_pose/predicted_contact"
            )
        if not float(tau) > 0.0:
            raise ValueError("tau must be positive")
        if not float(anchor_mass_epsilon) >= 0.0:
            raise ValueError("anchor_mass_epsilon must be non-negative")
        self.candidate = candidate
        self.tau = float(tau)
        self.anchor_mass_epsilon = float(anchor_mass_epsilon)
        self.detach_predicted = bool(detach_predicted)
        self._allow_future_ground_truth = allow_future_ground_truth is True

    def action_only(
        self,
        object_valid: torch.Tensor,
        timesteps: int,
        *,
        reference: torch.Tensor | None = None,
    ) -> ContactConditioningResult:
        if self.candidate != "metric_pose":
            raise RuntimeError("action_only is available only for metric_pose")
        return zero_contact_conditioning(
            object_valid, timesteps, reference=reference
        )

    def predicted(
        self,
        *,
        canonical_xyz: torch.Tensor,
        point_object_id: torch.Tensor,
        point_valid: torch.Tensor,
        object_valid: torch.Tensor,
        predicted_point_hand_distance: torch.Tensor,
        hand_valid: torch.Tensor,
    ) -> ContactConditioningResult:
        if self.candidate != "predicted_contact":
            raise RuntimeError("predicted is available only for predicted_contact")
        return predicted_contact_conditioning(
            canonical_xyz=canonical_xyz,
            point_object_id=point_object_id,
            point_valid=point_valid,
            object_valid=object_valid,
            predicted_point_hand_distance=predicted_point_hand_distance,
            hand_valid=hand_valid,
            tau=self.tau,
            anchor_mass_epsilon=self.anchor_mass_epsilon,
            detach=self.detach_predicted,
        )

    def oracle(
        self,
        *,
        canonical_xyz: torch.Tensor,
        point_object_id: torch.Tensor,
        object_valid: torch.Tensor,
        future_object_xyz: torch.Tensor,
        future_point_valid: torch.Tensor,
        hand_xyz: torch.Tensor,
        hand_valid: torch.Tensor,
        distance_chunk_bt: int = 32,
    ) -> ContactConditioningResult:
        if self.candidate != "oracle_contact":
            raise RuntimeError("oracle is available only for oracle_contact")
        return oracle_contact_conditioning(
            canonical_xyz=canonical_xyz,
            point_object_id=point_object_id,
            object_valid=object_valid,
            future_object_xyz=future_object_xyz,
            future_point_valid=future_point_valid,
            hand_xyz=hand_xyz,
            hand_valid=hand_valid,
            allow_future_ground_truth=self._allow_future_ground_truth,
            tau=self.tau,
            anchor_mass_epsilon=self.anchor_mass_epsilon,
            distance_chunk_bt=distance_chunk_bt,
        )
