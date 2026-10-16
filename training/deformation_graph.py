"""Canonical embedded-deformation graph geometry for long-horizon HOI.

This module contains no campaign or trainer state.  It implements the sparse
canonical graph warp introduced by Embedded Deformation and the associated
ARAP-style graph regularizer in ordinary PyTorch so it can be tested without
the PointTransformer backbone.

The graph is rebuilt deterministically from persistent canonical tracked
points on every forward pass.  Graph nodes and skinning weights therefore do
not depend on future geometry and cannot leak evaluation targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _require_shape(tensor: torch.Tensor, suffix: tuple[int, ...], name: str) -> None:
    if tensor.ndim < len(suffix) or tuple(tensor.shape[-len(suffix) :]) != suffix:
        raise ValueError(f"{name} must end in shape {suffix}, got {tuple(tensor.shape)}")


def deterministic_farthest_point_indices(
    points: torch.Tensor,
    count: int,
) -> torch.Tensor:
    """Select up to ``count`` unique points with deterministic FPS.

    The first node is the point farthest from the object centroid, avoiding an
    arbitrary fixed input index for ordinary (non-tied) shapes.  ``argmax``
    resolves exact ties by the stable input order.
    """

    _require_shape(points, (3,), "points")
    if points.ndim != 2:
        raise ValueError("points must have shape (N,3)")
    if count <= 0:
        raise ValueError("count must be positive")
    point_count = int(points.shape[0])
    if point_count == 0:
        return torch.empty(0, dtype=torch.long, device=points.device)
    selected_count = min(int(count), point_count)
    work = points.float()
    centroid = work.mean(dim=0)
    first = torch.linalg.vector_norm(work - centroid, dim=-1).argmax()
    selected = [first]
    minimum_squared_distance = (work - work[first]).square().sum(dim=-1)
    minimum_squared_distance[first] = -1.0
    for _ in range(1, selected_count):
        index = minimum_squared_distance.argmax()
        selected.append(index)
        squared_distance = (work - work[index]).square().sum(dim=-1)
        minimum_squared_distance = torch.minimum(
            minimum_squared_distance, squared_distance
        )
        minimum_squared_distance[torch.stack(selected)] = -1.0
    return torch.stack(selected).long()


@dataclass(frozen=True)
class DeformationGraphTopology:
    """Per-batch canonical graph and sparse skinning assignment."""

    node_centers: torch.Tensor
    node_valid: torch.Tensor
    point_node_indices: torch.Tensor
    point_node_weights: torch.Tensor
    node_neighbors: torch.Tensor
    node_neighbor_valid: torch.Tensor
    node_mass: torch.Tensor

    def validate(self) -> None:
        _require_shape(self.node_centers, (3,), "node_centers")
        if self.node_centers.ndim != 4:
            raise ValueError("node_centers must have shape (B,O,K,3)")
        batch, objects, nodes = self.node_centers.shape[:3]
        if tuple(self.node_valid.shape) != (batch, objects, nodes):
            raise ValueError("node_valid shape does not match node_centers")
        if tuple(self.node_mass.shape) != (batch, objects, nodes):
            raise ValueError("node_mass shape does not match node_centers")
        if self.point_node_indices.shape != self.point_node_weights.shape:
            raise ValueError("point-node index and weight shapes must match")
        if self.point_node_indices.ndim != 3 or self.point_node_indices.shape[0] != batch:
            raise ValueError("point-node tensors must have shape (B,N,J)")
        if self.node_neighbors.shape != self.node_neighbor_valid.shape:
            raise ValueError("node-neighbor index and valid shapes must match")
        if tuple(self.node_neighbors.shape[:3]) != (batch, objects, nodes):
            raise ValueError("node-neighbor shape does not match graph")
        if not torch.isfinite(self.node_centers).all():
            raise ValueError("node centers contain non-finite values")
        if not torch.isfinite(self.point_node_weights).all():
            raise ValueError("skinning weights contain non-finite values")
        if (self.point_node_weights < 0).any():
            raise ValueError("skinning weights must be non-negative")


def _node_radius(centers: torch.Tensor, radius_scale: float, minimum: float) -> torch.Tensor:
    if int(centers.shape[0]) <= 1:
        return centers.new_tensor(float(minimum))
    pairwise = torch.cdist(centers.float(), centers.float())
    pairwise.fill_diagonal_(float("inf"))
    nearest = pairwise.amin(dim=-1)
    radius = nearest.median() * float(radius_scale)
    return radius.clamp_min(float(minimum)).to(dtype=centers.dtype)


def build_deformation_graph(
    canonical_xyz: torch.Tensor,
    point_object_id: torch.Tensor,
    point_valid: torch.Tensor,
    object_valid: torch.Tensor,
    *,
    node_count: int,
    influence_count: int = 4,
    edge_count: int = 4,
    radius_scale: float = 1.5,
    minimum_radius_m: float = 0.005,
) -> DeformationGraphTopology:
    """Build a fixed-size canonical graph without looking at future frames."""

    _require_shape(canonical_xyz, (3,), "canonical_xyz")
    if canonical_xyz.ndim != 3:
        raise ValueError("canonical_xyz must have shape (B,N,3)")
    batch, point_count = canonical_xyz.shape[:2]
    if tuple(point_object_id.shape) != (batch, point_count):
        raise ValueError("point_object_id must have shape (B,N)")
    if tuple(point_valid.shape) != (batch, point_count):
        raise ValueError("point_valid must have shape (B,N)")
    if object_valid.ndim != 2 or object_valid.shape[0] != batch:
        raise ValueError("object_valid must have shape (B,O)")
    if node_count <= 0 or influence_count <= 0 or edge_count <= 0:
        raise ValueError("node, influence, and edge counts must be positive")
    if not math.isfinite(radius_scale) or radius_scale <= 0:
        raise ValueError("radius_scale must be finite and positive")
    if not math.isfinite(minimum_radius_m) or minimum_radius_m <= 0:
        raise ValueError("minimum_radius_m must be finite and positive")

    device = canonical_xyz.device
    dtype = canonical_xyz.dtype
    objects = int(object_valid.shape[1])
    neighbors_per_point = min(int(influence_count), int(node_count))
    edges_per_node = min(int(edge_count), max(int(node_count) - 1, 1))
    centers = canonical_xyz.new_zeros(batch, objects, node_count, 3)
    node_valid = torch.zeros(batch, objects, node_count, dtype=torch.bool, device=device)
    point_indices = torch.zeros(
        batch, point_count, neighbors_per_point, dtype=torch.long, device=device
    )
    point_weights = canonical_xyz.new_zeros(batch, point_count, neighbors_per_point)
    node_neighbors = torch.zeros(
        batch, objects, node_count, edges_per_node, dtype=torch.long, device=device
    )
    node_neighbor_valid = torch.zeros(
        batch, objects, node_count, edges_per_node, dtype=torch.bool, device=device
    )

    ids = point_object_id.long()
    valid_points = point_valid.bool()
    for batch_index in range(batch):
        for object_index in range(objects):
            mask = valid_points[batch_index] & (ids[batch_index] == object_index)
            if not bool(object_valid[batch_index, object_index]) or not mask.any():
                continue
            object_point_indices = mask.nonzero(as_tuple=False).squeeze(-1)
            object_points = canonical_xyz[batch_index, object_point_indices]
            chosen = deterministic_farthest_point_indices(object_points, node_count)
            active_nodes = int(chosen.numel())
            object_centers = object_points[chosen]
            centers[batch_index, object_index, :active_nodes] = object_centers
            node_valid[batch_index, object_index, :active_nodes] = True

            assignment_count = min(neighbors_per_point, active_nodes)
            distances = torch.cdist(object_points.float(), object_centers.float())
            nearest_distance, nearest_index = torch.topk(
                distances, assignment_count, dim=-1, largest=False, sorted=True
            )
            radius = _node_radius(object_centers, radius_scale, minimum_radius_m).float()
            weights = torch.exp(-0.5 * (nearest_distance / radius).square())
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            point_indices[
                batch_index, object_point_indices, :assignment_count
            ] = nearest_index
            point_weights[
                batch_index, object_point_indices, :assignment_count
            ] = weights.to(dtype=dtype)

            if active_nodes > 1:
                actual_edges = min(edges_per_node, active_nodes - 1)
                node_distance = torch.cdist(
                    object_centers.float(), object_centers.float()
                )
                node_distance.fill_diagonal_(float("inf"))
                nearest_nodes = torch.topk(
                    node_distance,
                    actual_edges,
                    dim=-1,
                    largest=False,
                    sorted=True,
                ).indices
                node_neighbors[
                    batch_index, object_index, :active_nodes, :actual_edges
                ] = nearest_nodes
                node_neighbor_valid[
                    batch_index, object_index, :active_nodes, :actual_edges
                ] = True

    node_mass = canonical_xyz.new_zeros(batch, objects, node_count)
    for batch_index in range(batch):
        for object_index in range(objects):
            mask = valid_points[batch_index] & (ids[batch_index] == object_index)
            if not mask.any():
                continue
            indices = point_indices[batch_index, mask]
            weights = point_weights[batch_index, mask]
            node_mass[batch_index, object_index].scatter_add_(
                0, indices.reshape(-1), weights.reshape(-1)
            )

    topology = DeformationGraphTopology(
        node_centers=centers,
        node_valid=node_valid,
        point_node_indices=point_indices,
        point_node_weights=point_weights,
        node_neighbors=node_neighbors,
        node_neighbor_valid=node_neighbor_valid,
        node_mass=node_mass,
    )
    topology.validate()
    return topology


def mass_center_node_twists(
    twists: torch.Tensor,
    node_mass: torch.Tensor,
    node_valid: torch.Tensor,
) -> torch.Tensor:
    """Remove the graph-wide mean twist so the global pose owns rigid motion."""

    _require_shape(twists, (6,), "twists")
    if twists.ndim != 5:
        raise ValueError("twists must have shape (B,O,T,K,6)")
    expected = twists.shape[:2] + twists.shape[3:4]
    if tuple(node_mass.shape) != tuple(expected):
        raise ValueError("node_mass must have shape (B,O,K)")
    if tuple(node_valid.shape) != tuple(expected):
        raise ValueError("node_valid must have shape (B,O,K)")
    mass = node_mass.to(dtype=twists.dtype) * node_valid.to(dtype=twists.dtype)
    mass = mass[:, :, None, :, None]
    denominator = mass.sum(dim=3, keepdim=True).clamp_min(1e-12)
    mean = (twists * mass).sum(dim=3, keepdim=True) / denominator
    centered = (twists - mean) * node_valid[:, :, None, :, None]
    return centered


def pool_point_features_to_nodes(
    point_features: torch.Tensor,
    point_object_id: torch.Tensor,
    point_valid: torch.Tensor,
    topology: DeformationGraphTopology,
) -> torch.Tensor:
    """Skinning-weighted pooling of semantic point features onto graph nodes."""

    if point_features.ndim != 3:
        raise ValueError("point_features must have shape (B,N,C)")
    batch, point_count, channels = point_features.shape
    objects, nodes = topology.node_centers.shape[1:3]
    if tuple(point_object_id.shape) != (batch, point_count):
        raise ValueError("point_object_id must have shape (B,N)")
    if tuple(point_valid.shape) != (batch, point_count):
        raise ValueError("point_valid must have shape (B,N)")
    pooled = point_features.new_zeros(batch, objects, nodes, channels)
    ids = point_object_id.long()
    valid = point_valid.bool()
    for batch_index in range(batch):
        for object_index in range(objects):
            mask = valid[batch_index] & (ids[batch_index] == object_index)
            if not mask.any():
                continue
            node_indices = topology.point_node_indices[batch_index, mask]
            skinning = topology.point_node_weights[batch_index, mask]
            features = point_features[batch_index, mask]
            contribution = features[:, None] * skinning[..., None]
            pooled[batch_index, object_index].scatter_add_(
                0,
                node_indices[..., None].expand(-1, -1, channels).reshape(
                    -1, channels
                ),
                contribution.reshape(-1, channels),
            )
    denominator = topology.node_mass.float()[..., None].clamp_min(1e-6)
    normalized = pooled.float() / denominator
    normalized = normalized * topology.node_valid[..., None]
    return normalized.to(dtype=point_features.dtype)


def weighted_kabsch_projection(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    fallback_rotation: torch.Tensor | None = None,
    fallback_translation: torch.Tensor | None = None,
    rank_epsilon: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit one proper rigid transform from ``source`` to every target frame.

    ``source`` is ``(N,3)`` and ``target`` is ``(T,N,3)``.  The SVD runs in
    float32 for AMP compatibility.  Rank-deficient fits use the supplied
    fallback transform and return ``fit_valid=False``.
    """

    _require_shape(source, (3,), "source")
    _require_shape(target, (3,), "target")
    if source.ndim != 2 or target.ndim != 3 or target.shape[1:] != source.shape:
        raise ValueError("source/target must have shape (N,3)/(T,N,3)")
    frame_count, point_count = target.shape[:2]
    if weights is None:
        weights = source.new_ones(point_count)
    if tuple(weights.shape) != (point_count,):
        raise ValueError("weights must have shape (N,)")
    if rank_epsilon <= 0 or not math.isfinite(rank_epsilon):
        raise ValueError("rank_epsilon must be finite and positive")
    weight = weights.float().clamp_min(0)
    weight_sum = weight.sum()
    source_f = source.float()
    target_f = target.float()
    source_centroid = (source_f * weight[:, None]).sum(dim=0) / weight_sum.clamp_min(1e-12)
    target_centroid = (
        target_f * weight[None, :, None]
    ).sum(dim=1) / weight_sum.clamp_min(1e-12)
    source_centered = source_f - source_centroid
    target_centered = target_f - target_centroid[:, None]
    with torch.autocast(device_type=source.device.type, enabled=False):
        # Under bf16 autocast the einsum would emit a bf16 covariance and
        # CUDA has no bf16 SVD ("svd_cuda_gesvdjBatched"); keep it float32.
        covariance = torch.einsum(
            "n,ni,tnj->tij", weight, source_centered, target_centered
        ).float()
        u, singular, vh = torch.linalg.svd(covariance, full_matrices=False)
        vu = vh.transpose(-1, -2) @ u.transpose(-1, -2)
        determinant = torch.linalg.det(vu)
        correction = torch.ones(frame_count, 3, device=source.device, dtype=torch.float32)
        correction[:, -1] = torch.where(
            determinant < 0,
            -torch.ones_like(determinant),
            torch.ones_like(determinant),
        )
        rotation = vh.transpose(-1, -2) @ torch.diag_embed(correction) @ u.transpose(-1, -2)
        translation = target_centroid - torch.einsum(
            "tij,j->ti", rotation, source_centroid
        )
    enough_points = point_count >= 3 and bool(weight_sum > 0)
    fit_valid = torch.isfinite(rotation).all(dim=(-2, -1)) & torch.isfinite(
        translation
    ).all(dim=-1)
    if enough_points:
        fit_valid = fit_valid & (singular[:, 1] > float(rank_epsilon))
    else:
        fit_valid = torch.zeros_like(fit_valid)

    dtype = target.dtype
    rotation = rotation.to(dtype=dtype)
    translation = translation.to(dtype=dtype)
    if fallback_rotation is None:
        fallback_rotation = torch.eye(3, device=target.device, dtype=dtype).expand(
            frame_count, -1, -1
        )
    if fallback_translation is None:
        fallback_translation = target_centroid.to(dtype=dtype) - source_centroid.to(
            dtype=dtype
        )
    if tuple(fallback_rotation.shape) != (frame_count, 3, 3):
        raise ValueError("fallback_rotation must have shape (T,3,3)")
    if tuple(fallback_translation.shape) != (frame_count, 3):
        raise ValueError("fallback_translation must have shape (T,3)")
    rotation = torch.where(
        fit_valid[:, None, None], rotation, fallback_rotation.to(dtype=dtype)
    )
    translation = torch.where(
        fit_valid[:, None], translation, fallback_translation.to(dtype=dtype)
    )
    return rotation, translation, fit_valid


def warp_points_with_deformation_graph(
    canonical_xyz: torch.Tensor,
    point_object_id: torch.Tensor,
    point_valid: torch.Tensor,
    topology: DeformationGraphTopology,
    local_rotation: torch.Tensor,
    local_translation: torch.Tensor,
    global_rotation: torch.Tensor,
    global_translation: torch.Tensor,
    *,
    fallback_points: torch.Tensor,
) -> torch.Tensor:
    """Apply the embedded graph warp and then the global object pose."""

    topology.validate()
    batch, point_count = canonical_xyz.shape[:2]
    objects, nodes = topology.node_centers.shape[1:3]
    if local_rotation.ndim != 6 or tuple(local_rotation.shape[:2]) != (batch, objects):
        raise ValueError("local_rotation must have shape (B,O,T,K,3,3)")
    frame_count = int(local_rotation.shape[2])
    if tuple(local_rotation.shape[3:]) != (nodes, 3, 3):
        raise ValueError("local_rotation node shape mismatch")
    if tuple(local_translation.shape) != (batch, objects, frame_count, nodes, 3):
        raise ValueError("local_translation shape mismatch")
    if tuple(global_rotation.shape) != (batch, objects, frame_count, 3, 3):
        raise ValueError("global_rotation shape mismatch")
    if tuple(global_translation.shape) != (batch, objects, frame_count, 3):
        raise ValueError("global_translation shape mismatch")
    if tuple(fallback_points.shape) != (batch, frame_count, point_count, 3):
        raise ValueError("fallback_points must have shape (B,T,N,3)")

    output = fallback_points.clone()
    ids = point_object_id.long()
    valid = point_valid.bool()
    for batch_index in range(batch):
        for object_index in range(objects):
            mask = valid[batch_index] & (ids[batch_index] == object_index)
            if not mask.any() or not topology.node_valid[batch_index, object_index].any():
                continue
            points = canonical_xyz[batch_index, mask]
            indices = topology.point_node_indices[batch_index, mask]
            weights = topology.point_node_weights[batch_index, mask]
            centers = topology.node_centers[batch_index, object_index]
            selected_centers = centers[indices].to(dtype=canonical_xyz.dtype)
            # Index the node axis after slicing batch/object.  Keeping the
            # basic ``:`` and advanced tensor index adjacent this way gives
            # the intentional (T,N,J,...) layout on every supported PyTorch
            # version.
            selected_rotation = local_rotation[batch_index, object_index][
                :, indices
            ].to(dtype=canonical_xyz.dtype)
            selected_translation = local_translation[batch_index, object_index][
                :, indices
            ].to(dtype=canonical_xyz.dtype)
            relative = points[:, None] - selected_centers
            locally_warped = torch.einsum(
                "tnkij,nkj->tnki", selected_rotation, relative
            ) + selected_centers[None] + selected_translation
            deformed = (
                locally_warped * weights[None, :, :, None]
            ).sum(dim=2)
            world = torch.einsum(
                "tij,tnj->tni",
                global_rotation[batch_index, object_index].to(
                    dtype=canonical_xyz.dtype
                ),
                deformed,
            ) + global_translation[batch_index, object_index, :, None].to(
                dtype=canonical_xyz.dtype
            )
            output[batch_index, :, mask] = world
    if not torch.isfinite(output).all():
        raise FloatingPointError("deformation-graph warp produced non-finite points")
    return output


def arap_graph_loss(
    topology: DeformationGraphTopology,
    local_rotation: torch.Tensor,
    local_translation: torch.Tensor,
    *,
    epsilon_m: float = 1e-4,
) -> torch.Tensor:
    """Embedded-deformation edge consistency in metres (smooth L2)."""

    batch, objects, frames, nodes = local_translation.shape[:4]
    total = local_translation.sum() * 0.0
    count = local_translation.new_zeros(())
    for batch_index in range(batch):
        for object_index in range(objects):
            centers = topology.node_centers[batch_index, object_index].to(
                dtype=local_rotation.dtype
            )
            for node_index in range(nodes):
                valid_edges = topology.node_neighbor_valid[
                    batch_index, object_index, node_index
                ]
                if not valid_edges.any():
                    continue
                neighbor = topology.node_neighbors[
                    batch_index, object_index, node_index, valid_edges
                ]
                center = centers[node_index]
                neighbor_center = centers[neighbor]
                by_source = torch.einsum(
                    "tij,ej->tei",
                    local_rotation[batch_index, object_index, :, node_index],
                    neighbor_center - center,
                ) + center + local_translation[
                    batch_index, object_index, :, node_index, None
                ]
                by_neighbor = neighbor_center[None] + local_translation[
                    batch_index, object_index
                ][:, neighbor]
                difference = by_source - by_neighbor
                distance = torch.sqrt(
                    difference.square().sum(dim=-1) + float(epsilon_m) ** 2
                ) - float(epsilon_m)
                total = total + distance.sum()
                count = count + distance.numel()
    return total / count.clamp_min(1.0)


def local_transform_magnitude(
    local_translation: torch.Tensor,
    local_rotation_vector: torch.Tensor,
    node_valid: torch.Tensor,
    *,
    rotation_length_scale_m: float = 0.02,
) -> torch.Tensor:
    """Mean local-transform magnitude in equivalent metres."""

    valid = node_valid[:, :, None].expand(local_translation.shape[:-1])
    magnitude = torch.linalg.vector_norm(local_translation, dim=-1) + float(
        rotation_length_scale_m
    ) * torch.linalg.vector_norm(local_rotation_vector, dim=-1)
    return magnitude[valid].mean() if valid.any() else magnitude.sum() * 0.0


def temporal_transform_acceleration(
    local_translation: torch.Tensor,
    local_rotation_vector: torch.Tensor,
    node_valid: torch.Tensor,
    *,
    rotation_length_scale_m: float = 0.02,
) -> torch.Tensor:
    """Mean second temporal difference of local graph transforms."""

    if int(local_translation.shape[2]) < 3:
        return local_translation.sum() * 0.0
    translation_second = (
        local_translation[:, :, 2:]
        - 2.0 * local_translation[:, :, 1:-1]
        + local_translation[:, :, :-2]
    )
    rotation_second = (
        local_rotation_vector[:, :, 2:]
        - 2.0 * local_rotation_vector[:, :, 1:-1]
        + local_rotation_vector[:, :, :-2]
    )
    magnitude = torch.linalg.vector_norm(translation_second, dim=-1) + float(
        rotation_length_scale_m
    ) * torch.linalg.vector_norm(rotation_second, dim=-1)
    valid = node_valid[:, :, None].expand(magnitude.shape)
    return magnitude[valid].mean() if valid.any() else magnitude.sum() * 0.0


__all__ = [
    "DeformationGraphTopology",
    "arap_graph_loss",
    "build_deformation_graph",
    "deterministic_farthest_point_indices",
    "local_transform_magnitude",
    "mass_center_node_twists",
    "pool_point_features_to_nodes",
    "temporal_transform_acceleration",
    "warp_points_with_deformation_graph",
    "weighted_kabsch_projection",
]
