"""Low-rank graph/time bases and local hand signals for deformation forecasting."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from training.carrier_gauge_core import autocast_disabled


def clamped_bspline_basis(
    length: int,
    controls: int,
    degree: int = 3,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Open-uniform clamped B-spline basis ``(length, controls)``."""
    if length <= 0 or degree < 0 or controls < degree + 1:
        raise ValueError("invalid B-spline dimensions")
    interior = controls - degree - 1
    knots = [0.0] * (degree + 1)
    knots += [i / (interior + 1) for i in range(1, interior + 1)]
    knots += [1.0] * (degree + 1)
    knot = torch.tensor(knots, device=device, dtype=torch.float64)
    parameter = torch.linspace(
        0, 1, length, device=device, dtype=torch.float64
    )
    basis = torch.zeros(
        length, controls + degree, device=device, dtype=torch.float64
    )
    for i in range(controls + degree):
        basis[:, i] = (
            (parameter >= knot[i]) & (parameter < knot[i + 1])
        ).to(basis.dtype)
    basis[-1].zero_()
    basis[-1, controls - 1] = 1
    for d in range(1, degree + 1):
        updated = torch.zeros_like(basis)
        for i in range(controls + degree - d):
            left_d = knot[i + d] - knot[i]
            right_d = knot[i + d + 1] - knot[i + 1]
            if bool(left_d > 0):
                updated[:, i] += (
                    (parameter - knot[i]) / left_d * basis[:, i]
                )
            if bool(right_d > 0):
                updated[:, i] += (
                    (knot[i + d + 1] - parameter)
                    / right_d
                    * basis[:, i + 1]
                )
        basis = updated
    basis = basis[:, :controls]
    return (
        basis / basis.sum(-1, keepdim=True).clamp_min(1e-12)
    ).to(dtype)


@autocast_disabled
def graph_laplacian_basis(topology: Any, rank: int) -> torch.Tensor:
    """Deterministic low-frequency nonconstant modes ``(B,O,K,R)``."""
    centers = topology.node_centers
    node_valid = topology.node_valid.bool()
    batch, objects, nodes = centers.shape[:3]
    output = centers.new_zeros(batch, objects, nodes, rank)
    for b in range(batch):
        for o in range(objects):
            active = node_valid[b, o].nonzero(as_tuple=False).squeeze(-1)
            count = int(active.numel())
            if count <= 1:
                continue
            local = {
                int(node): i for i, node in enumerate(active.tolist())
            }
            adjacency = torch.zeros(
                count, count, device=centers.device, dtype=torch.float32
            )
            points = centers[b, o, active].float()
            edges: list[tuple[int, int]] = []
            distances: list[torch.Tensor] = []
            for source in active.tolist():
                for edge in range(topology.node_neighbors.shape[-1]):
                    if not bool(
                        topology.node_neighbor_valid[b, o, source, edge]
                    ):
                        continue
                    destination = int(
                        topology.node_neighbors[b, o, source, edge]
                    )
                    if destination in local and destination != source:
                        i = local[source]
                        j = local[destination]
                        edges.append((i, j))
                        distances.append(
                            torch.linalg.vector_norm(points[i] - points[j])
                        )
            if distances:
                sigma = torch.stack(distances).median().clamp_min(1e-4)
                for (i, j), distance in zip(edges, distances):
                    value = torch.exp(
                        -0.5 * (distance / sigma).square()
                    )
                    adjacency[i, j] = torch.maximum(
                        adjacency[i, j], value
                    )
                    adjacency[j, i] = torch.maximum(
                        adjacency[j, i], value
                    )
            degree = adjacency.sum(-1)
            laplacian = torch.diag(degree) - adjacency
            _, eigenvectors = torch.linalg.eigh(laplacian)
            available = min(rank, count - 1)
            selected = eigenvectors[:, 1 : 1 + available]
            if available:
                pivot = selected.abs().argmax(0)
                sign = torch.sign(
                    selected[
                        pivot,
                        torch.arange(available, device=centers.device),
                    ]
                )
                selected = selected * torch.where(
                    sign == 0, torch.ones_like(sign), sign
                )[None]
                output[b, o, active, :available] = selected.to(output.dtype)
    return output


@autocast_disabled
def temporal_project(
    signal: torch.Tensor, basis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares project ``(...,T,R,C)`` into a temporal basis."""
    dtype = torch.promote_types(signal.dtype, basis.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    basis_f = basis.to(signal.device, dtype)
    signal_f = signal.to(dtype)
    coefficients = torch.einsum(
        "qt,...trc->...qrc", torch.linalg.pinv(basis_f), signal_f
    )
    reconstructed = torch.einsum(
        "tq,...qrc->...trc", basis_f, coefficients
    )
    return reconstructed.to(signal.dtype), coefficients.to(signal.dtype)


def node_hand_features(
    node_centers: torch.Tensor,
    node_valid: torch.Tensor,
    initial_rotation: torch.Tensor,
    initial_translation: torch.Tensor,
    hand_xyz: torch.Tensor,
    hand_valid: torch.Tensor,
    *,
    temperature_m: float = 0.04,
) -> torch.Tensor:
    """Per-node future left/right soft-nearest relations; output width 18."""
    if hand_valid.ndim == hand_xyz.ndim:
        hand_valid = hand_valid.squeeze(-1)
    _, _, hand_points = hand_xyz.shape[:3]
    if (
        hand_points % 2
        or not math.isfinite(temperature_m)
        or temperature_m <= 0
    ):
        raise ValueError("invalid two-hand representation or temperature")
    node_world = (
        torch.einsum(
            "boij,bokj->boki", initial_rotation, node_centers
        )
        + initial_translation[:, :, None]
    )
    world_to_object = initial_rotation.transpose(-1, -2)
    half = hand_points // 2
    features = []
    for side in range(2):
        xyz = hand_xyz[:, :, side * half : (side + 1) * half]
        valid = hand_valid[:, :, side * half : (side + 1) * half].bool()
        relative_world = (
            xyz[:, None, :, None] - node_world[:, :, None, :, None]
        )
        relative = torch.einsum(
            "boij,botkhj->botkhi", world_to_object, relative_world
        )
        distance = torch.linalg.vector_norm(relative.float(), dim=-1)
        support = (
            valid[:, None, :, None]
            & node_valid[:, :, None, :, None].bool()
        )
        weight = (
            torch.exp(-0.5 * (distance / temperature_m).square())
            * support.float()
        )
        total = weight.sum(-1, keepdim=True).clamp_min(1e-12)
        soft = (relative.float() * weight[..., None]).sum(-2) / total
        present = support.any(-1)
        soft = torch.where(
            present[..., None], soft, torch.zeros_like(soft)
        )
        delta = torch.zeros_like(soft)
        delta[:, :, 1:] = soft[:, :, 1:] - soft[:, :, :-1]
        minimum = distance.masked_fill(~support, float("inf")).amin(-1)
        minimum = torch.where(present, minimum, torch.ones_like(minimum))
        side_feature = torch.cat(
            (
                soft,
                delta,
                weight.amax(-1)[..., None],
                minimum[..., None],
                present[..., None].to(soft.dtype),
            ),
            -1,
        )[:, :, 1:]
        features.append(side_feature)
    return torch.cat(features, -1).to(node_centers.dtype)


def edge_strain_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    canonical: torch.Tensor,
    object_ids: torch.Tensor,
    point_valid: torch.Tensor,
    topology: Any,
    *,
    future_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match predicted graph-edge lengths to GT, permitting real deformation."""
    total = predicted.sum() * 0
    count = torch.zeros((), device=predicted.device, dtype=torch.long)
    for b in range(predicted.shape[0]):
        for o in range(topology.node_centers.shape[1]):
            mask = point_valid[b].bool() & (object_ids[b].long() == o)
            points = mask.nonzero(as_tuple=False).squeeze(-1)
            active = topology.node_valid[b, o].nonzero(
                as_tuple=False
            ).squeeze(-1)
            if int(points.numel()) < 2 or int(active.numel()) < 2:
                continue
            nearest = torch.cdist(
                topology.node_centers[b, o, active].float(),
                canonical[b, points].float(),
            ).argmin(-1)
            representative = {
                int(node): int(points[nearest[i]])
                for i, node in enumerate(active.tolist())
            }
            pairs: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for source in active.tolist():
                for edge in range(topology.node_neighbors.shape[-1]):
                    if not bool(
                        topology.node_neighbor_valid[b, o, source, edge]
                    ):
                        continue
                    destination = int(
                        topology.node_neighbors[b, o, source, edge]
                    )
                    key = tuple(sorted((source, destination)))
                    if (
                        destination in representative
                        and key[0] != key[1]
                        and key not in seen
                    ):
                        seen.add(key)
                        pairs.append(
                            (
                                representative[key[0]],
                                representative[key[1]],
                            )
                        )
            if not pairs:
                continue
            source = torch.tensor(
                [pair[0] for pair in pairs], device=predicted.device
            )
            destination = torch.tensor(
                [pair[1] for pair in pairs], device=predicted.device
            )
            pred_length = torch.linalg.vector_norm(
                predicted[b, 1:, source]
                - predicted[b, 1:, destination],
                dim=-1,
            )
            gt_length = torch.linalg.vector_norm(
                target[b, 1:, source] - target[b, 1:, destination],
                dim=-1,
            )
            loss = F.smooth_l1_loss(
                pred_length, gt_length, reduction="none", beta=0.005
            )
            if future_valid is not None:
                valid = (
                    future_valid[b, 1:, source].bool()
                    & future_valid[b, 1:, destination].bool()
                )
                total += loss[valid].sum()
                count += valid.sum()
            else:
                total += loss.sum()
                count += loss.numel()
    return total / count.clamp_min(1).to(total.dtype)


__all__ = [
    "clamped_bspline_basis",
    "edge_strain_loss",
    "graph_laplacian_basis",
    "node_hand_features",
    "temporal_project",
]
