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
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
import yaml
from ptv3.ptv3 import PointTransformerV3, MLP
from scene_featurizer import SceneFeatureEncoder
from pointworld.embeddings import SpatialEmbedding, TemporalEmbedding
from pointworld import norm_stats as norm_stats_utils
from pointworld import losses as losses_utils
from pointworld import metrics as metrics_utils
from utils import handle_nan_outputs

UNCERTAINTY_LOGVAR_WEIGHT = 1.0
SIM_VAR_CONST = 1e-3
CONTEXT_HORIZON = 1
PRED_HORIZON = 10
VAR_FLOOR = 1e-6
VAR_CEILING = 1e2
SIM_DOMAIN_KEYWORDS = ["behavior"]
PTV3_DROP_PATH = 0.3
_PTV3_BLUEPRINT = None
MANO_PARENT_21 = (-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19)
MANO_PARAM_KEYS = (
    "mano_pose_coeffs",
    "mano_hand_pose",
    "mano_thetas",
    "mano_betas",
    "mano_wrist_xform",
    "mano_wrist_transform",
    "mano_global_orient",
    "mano_translation",
    "mano_transl",
    "mano_translation_xyz",
    "mano_rot6d",
    "mano_valid",
)


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    orig_shape = axis_angle.shape[:-1]
    vec = axis_angle.reshape(-1, 3)
    angle = torch.linalg.norm(vec, dim=-1, keepdim=True)
    axis = vec / angle.clamp_min(1e-8)
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    k = torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    eye = torch.eye(3, device=vec.device, dtype=vec.dtype).expand(vec.shape[0], 3, 3)
    sin = torch.sin(angle).view(-1, 1, 1)
    cos = torch.cos(angle).view(-1, 1, 1)
    matrix = eye + sin * k + (1.0 - cos) * (k @ k)
    small = (angle.view(-1) < 1e-7).view(-1, 1, 1)
    matrix = torch.where(small, eye, matrix)
    return matrix.reshape(*orig_shape, 3, 3)


def _rotation_6d_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :3, :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6)


def _identity_rotation_matrix(batch_shape, device, dtype) -> torch.Tensor:
    return torch.eye(3, device=device, dtype=dtype).expand(*batch_shape, 3, 3).clone()


def _resolve_model_timesteps(args) -> int:
    if getattr(args, "dataset_type", "pointworld_wds") == "hoi_npz":
        timesteps = int(getattr(args, "hoi_target_horizon", CONTEXT_HORIZON + PRED_HORIZON))
    else:
        timesteps = CONTEXT_HORIZON + PRED_HORIZON
    if timesteps < CONTEXT_HORIZON + 1:
        raise ValueError(
            f"Model horizon must be at least {CONTEXT_HORIZON + 1} "
            f"({CONTEXT_HORIZON} context + >=1 prediction), got {timesteps}"
        )
    return timesteps


def _load_ptv3_blueprint():
    global _PTV3_BLUEPRINT
    if _PTV3_BLUEPRINT is not None:
        return _PTV3_BLUEPRINT
    blueprint_path = Path(__file__).resolve().parent.parent / "ptv3" / "ptv3_arch.yaml"
    if not blueprint_path.is_file():
        raise FileNotFoundError(f"PTV3 blueprint not found: {blueprint_path}")
    with open(blueprint_path, "r") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "sizes" not in data:
        raise ValueError("ptv3_arch.yaml must contain a top-level 'sizes' mapping")
    sizes = data["sizes"]
    if not isinstance(sizes, dict) or not sizes:
        raise ValueError("ptv3_arch.yaml 'sizes' must be a non-empty mapping")
    _PTV3_BLUEPRINT = sizes
    return _PTV3_BLUEPRINT


def _prepare_log_var_for_confidence(log_var, domains):
    if log_var is None:
        raise RuntimeError("log_var is required to compute confidence.")
    if not isinstance(domains, (list, tuple)):
        raise TypeError("domains must be a list or tuple of domain strings.")
    if log_var.ndim != 4:
        raise ValueError(f"log_var must be 4D (B,T,Ns,C); got shape {tuple(log_var.shape)}")
    B = log_var.shape[0]
    if len(domains) != B:
        raise ValueError(f"Domain list length ({len(domains)}) must match batch size ({B}).")

    sim_b = torch.tensor(
        [any(k in dom for k in SIM_DOMAIN_KEYWORDS) for dom in domains],
        device=log_var.device,
        dtype=torch.bool,
    )
    if sim_b.any():
        sim_mask = sim_b.view(B, 1, 1, 1).expand_as(log_var)
        real_b = ~sim_b
        if real_b.any():
            s_min_tmp = math.log(VAR_FLOOR)
            s_max_tmp = math.log(VAR_CEILING)
            clamped = log_var.detach().clamp(min=s_min_tmp, max=s_max_tmp)
            real_logvar = clamped[real_b]
            real_var_mean = real_logvar.exp().mean().clamp(
                min=float(VAR_FLOOR), max=float(VAR_CEILING)
            )
            const_val = float(real_var_mean.log().item())
        else:
            const_val = math.log(SIM_VAR_CONST)
        const_tensor = torch.full_like(log_var, fill_value=const_val)
        log_var = torch.where(sim_mask, const_tensor, log_var)

    s_min = math.log(VAR_FLOOR)
    s_max = math.log(VAR_CEILING)
    return log_var.clamp(min=s_min, max=s_max)


def _resolve_arch_config(size: str) -> dict:
    sizes = _load_ptv3_blueprint()
    if size not in sizes:
        raise ValueError(f"Invalid model args.ptv3_size: {size}. Available: {sorted(sizes.keys())}")
    cfg = sizes[size]
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config for size '{size}': expected mapping")
    alias = cfg.get("alias_of")
    if alias is not None:
        if alias not in sizes:
            raise ValueError(f"Invalid alias_of '{alias}' for size '{size}'")
        cfg = sizes[alias]
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid alias config for size '{alias}': expected mapping")
    return cfg


def _require_list(cfg: dict, key: str) -> list:
    if key not in cfg:
        raise KeyError(f"PTV3 blueprint missing required field '{key}'")
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"PTV3 blueprint '{key}' must be a list")
    return value


def _resolve_channels(items: list, channels: int) -> tuple:
    out = []
    for item in items:
        if isinstance(item, str):
            if item != "channels":
                raise ValueError(f"Unsupported placeholder '{item}' in PTV3 blueprint")
            out.append(int(channels))
        else:
            out.append(int(item))
    return tuple(out)


def _resolve_patch_size(value, depth_len: int, patch_size: int) -> tuple:
    if value is None or value == "auto":
        return (int(patch_size),) * depth_len
    if isinstance(value, list):
        if len(value) != depth_len:
            raise ValueError("PTV3 patch size list must match depth length")
        return tuple(int(v) for v in value)
    raise TypeError("PTV3 patch size must be 'auto' or a list of ints")


def _resolve_stride(value, enc_depths: tuple) -> tuple:
    if value is None or value == "auto":
        return (2,) * (len(enc_depths) - 1)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    raise TypeError("PTV3 stride must be 'auto' or a list of ints")


def build_ptv3(channels, args):
    cfg = _resolve_arch_config(args.ptv3_size)
    if "channels_max" in cfg:
        assert channels <= int(cfg["channels_max"]), (
            f"it's recommended to use a decoder dimension of {cfg['channels_max']} or less"
        )
    if "channels_eq" in cfg:
        assert channels == int(cfg["channels_eq"]), (
            f"it's recommended to use a decoder dimension of {cfg['channels_eq']}"
        )

    enc_depths = tuple(_require_list(cfg, "enc_depths"))
    dec_depths = tuple(_require_list(cfg, "dec_depths"))
    enc_channels = _resolve_channels(_require_list(cfg, "enc_channels"), channels)
    dec_channels = _resolve_channels(_require_list(cfg, "dec_channels"), channels)
    enc_num_head = tuple(int(v) for v in _require_list(cfg, "enc_num_head"))
    dec_num_head = tuple(int(v) for v in _require_list(cfg, "dec_num_head"))
    stride = _resolve_stride(cfg.get("stride", "auto"), enc_depths)
    enc_patch_size = _resolve_patch_size(cfg.get("enc_patch_size", "auto"), len(enc_depths), args.ptv3_patch_size)
    dec_patch_size = _resolve_patch_size(cfg.get("dec_patch_size", "auto"), len(dec_depths), args.ptv3_patch_size)

    # Create the PointTransformerV3 with the configured parameters
    # attention registers are disabled in release configuration
    attn_cls = None
    attn_kwargs = {}

    return PointTransformerV3(
        in_channels=channels,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=stride,
        enc_depths=enc_depths,
        enc_channels=enc_channels,
        enc_num_head=enc_num_head,
        enc_patch_size=enc_patch_size,
        dec_depths=dec_depths,
        dec_channels=dec_channels,
        dec_num_head=dec_num_head,
        dec_patch_size=dec_patch_size,
        mlp_ratio=4,
        qkv_bias=True,
        attn_drop=0,
        proj_drop=0,
        drop_path=PTV3_DROP_PATH,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        traceable=True,
        mask_token=False,
        freeze_encoder=False,
        enc_mode=False,
        attention_cls=attn_cls,
        attention_kwargs=attn_kwargs,
    )


class DynamicsPredictor(nn.Module):
    def __init__(self, args, in_channels, T):
        super().__init__()
        self.args, self.T, self.in_c = args, T, in_channels

        # Build the dynamics decoder based on size configuration
        self.predictor_model = build_ptv3(self.in_c, args)

        # Only predict T-1 timesteps since first timestep is input
        self.dynamics_head = nn.Sequential(MLP(in_channels, 128, 128), nn.Linear(128, 3 * (T-1)))
        with torch.no_grad():
            nn.init.kaiming_normal_(self.dynamics_head[-1].weight)
            self.dynamics_head[-1].weight.mul_(args.dynamics_head_init_scale)
            self.dynamics_head[-1].bias.zero_()

        # Add heteroscedastic uncertainty head - only for T-1 steps
        self.u_dim = 1
        self.log_var_head = nn.Sequential(
            MLP(in_channels, 128, 128),
            nn.Linear(128, self.u_dim * (T-1))
        )
        with torch.no_grad():
            self.log_var_head[-1].bias.fill_(math.log(0.005 ** 2))

        self.use_move_prediction_head = bool(getattr(args, "use_move_prediction_head", False))
        if self.use_move_prediction_head:
            self.move_head = nn.Sequential(MLP(in_channels, 128, 128), nn.Linear(128, T-1))
            with torch.no_grad():
                nn.init.kaiming_normal_(self.move_head[-1].weight)
                self.move_head[-1].weight.mul_(1e-3)
                self.move_head[-1].bias.zero_()

        # Skip connection FiLM parameters for scene features
        self.skip_film_gamma = nn.Parameter(torch.ones(1, 1, in_channels))
        self.skip_film_beta = nn.Parameter(torch.zeros(1, 1, in_channels))

        # Robot global summary FiLM parameters
        self.robot_film_gamma = nn.Parameter(torch.ones(1, 1, in_channels))
        self.robot_film_beta = nn.Parameter(torch.zeros(1, 1, in_channels))

        self.register_buffer("_grid_size", torch.tensor([args.grid_size]), persistent=False)

    def forward(self, scene_coord0, scene_feat0, scene_exists0, robot_coord_seq, robot_feat, robot_exists, normalize_fn=None, unnormalize_fn=None, training=True):
        B, Ns, _ = scene_coord0.shape
        T, Nr = robot_coord_seq.shape[1:3]
        if T != self.T:
            raise ValueError(f"DynamicsPredictor expected {self.T} timesteps, got {T}")
        device = scene_coord0.device

        coord = torch.cat([scene_coord0, robot_coord_seq.reshape(B, T*Nr, 3)], dim=1)  # (B, Ns+T*Nr, 3)
        feat  = torch.cat([scene_feat0,  robot_feat.reshape(B, T*Nr, robot_feat.shape[-1])], dim=1)  # (B, Ns+T*Nr, D)
        exists = torch.cat([scene_exists0, robot_exists.reshape(B, T*Nr)], dim=1)  # (B, Ns+T*Nr)
        is_robot = torch.cat([
            torch.zeros(B, Ns, 1, dtype=torch.bool, device=device),
            torch.ones (B, T*Nr,1, dtype=torch.bool, device=device)
        ], dim=1).flatten()  # (B*(Ns+T*Nr),)
        batch = torch.arange(B, device=device).repeat_interleave(Ns + T*Nr)  # (B*(Ns+T*Nr),)

        # Create data dictionary for transformer
        data_dict = {
            "coord": coord[exists],
            "feat" : feat[exists],
            "batch": batch[exists.flatten()],
            "is_robot": is_robot[exists.flatten()],
            "grid_size": self._grid_size,
        }

        point = self.predictor_model(data_dict)

        # Extract features with skip connection and robot global summary
        with torch.autocast('cuda', enabled=False):
            # Extract scene and robot features separately
            scene_mask = ~is_robot[exists.flatten()]
            robot_mask = is_robot[exists.flatten()]

            scene_feat_output = point.feat.new_zeros(B, Ns, self.in_c)  # (B,Ns,D)
            scene_feat_output[scene_exists0] = point.feat[scene_mask]  # (B,Ns,D)

            # Apply skip connection with FiLM modulation for input scene features
            skip_modulated = scene_feat0 * self.skip_film_gamma + self.skip_film_beta

            # Assert that we have robot features when not ignoring robot
            assert robot_mask.any(), "Expected robot features when not ignoring robot, but robot_mask is empty"

            # Create padded robot features tensor and fill with actual features
            padded_robot_features = torch.zeros(B, T*Nr, self.in_c, device=point.feat.device)  # (B, T*Nr, D)
            padded_robot_features[robot_exists.reshape(B, T*Nr)] = point.feat[robot_mask]  # Fill valid positions

            # Reshape to (B, T, Nr, D) for easier processing
            padded_robot_features = padded_robot_features.view(B, T, Nr, self.in_c)

            # Compute global robot summary for each batch
            robot_global_summaries = []
            for b in range(B):
                # Get valid robot features for this batch across all timesteps
                batch_robot_exists = robot_exists[b]  # (T, Nr)
                batch_robot_features = padded_robot_features[b]  # (T, Nr, D)

                # Extract only valid robot features
                valid_robot_features = batch_robot_features[batch_robot_exists]  # (N_valid_robot, D)

                if valid_robot_features.shape[0] > 0:
            # Max pooling across all valid robot points for this batch
                    batch_summary = valid_robot_features.max(dim=0, keepdim=True)[0]  # (1, D)
                else:
                    # If no robot features in this batch, use zero
                    batch_summary = torch.zeros(1, self.in_c, device=point.feat.device)

                robot_global_summaries.append(batch_summary)

            robot_global_summary = torch.stack(robot_global_summaries, dim=0)  # (B, 1, D)
            robot_global_summary = robot_global_summary.expand(-1, Ns, -1)  # (B, Ns, D)
            # Apply FiLM modulation for robot global summary
            robot_modulated = robot_global_summary * self.robot_film_gamma + self.robot_film_beta
            padded_scene_feat = scene_feat_output + skip_modulated + robot_modulated

            # dynamics head - now only for T-1 steps
            dynamics_head_output = self.dynamics_head(padded_scene_feat)  # (B,Ns,3*(T-1))
            dynamics_partial = dynamics_head_output.view(B, Ns, self.T-1, 3)  # (B,Ns,T-1,3)
            dynamics_partial = dynamics_partial.permute(0,2,1,3)  # (B,T-1,Ns,3)

            # Add zeros for first timestep
            zeros_first = torch.zeros(B, 1, Ns, 3, device=dynamics_partial.device, dtype=dynamics_partial.dtype)
            dynamics = torch.cat([zeros_first, dynamics_partial], dim=1)  # (B,T,Ns,3)

            # uncertainty head - now only for T-1 steps
            log_var_output = self.log_var_head(padded_scene_feat)  # (B,Ns,u_dim*(T-1))
            log_var_partial = log_var_output.view(B, Ns, self.T-1, self.u_dim)  # (B,Ns,T-1,1)
            log_var_partial = log_var_partial.permute(0,2,1,3)  # (B,T-1,Ns,1)

            zeros_first_var = torch.zeros(B, 1, Ns, self.u_dim, device=log_var_partial.device, dtype=log_var_partial.dtype)
            log_var = torch.cat([zeros_first_var, log_var_partial], dim=1)  # (B,T,Ns,1)

            outputs = {"pred": dynamics, "log_var": log_var}

            if self.use_move_prediction_head:
                move_logits_output = self.move_head(padded_scene_feat)  # (B,Ns,T-1)
                move_logits_partial = move_logits_output.permute(0, 2, 1)  # (B,T-1,Ns)
                zeros_first_move = torch.zeros(B, 1, Ns, device=move_logits_partial.device, dtype=move_logits_partial.dtype)
                outputs["move_logits"] = torch.cat([zeros_first_move, move_logits_partial], dim=1)  # (B,T,Ns)

        return outputs


def _mano_parent_indices(num_joints: int, device) -> torch.Tensor:
    if num_joints <= len(MANO_PARENT_21):
        parents = list(MANO_PARENT_21[:num_joints])
    else:
        # Keep the standard MANO-21 tree for the first 21 joints and attach any
        # extra implementation-specific joints to the wrist. This is intentionally
        # conservative; server-side agents should replace it if their MANO export
        # uses a known extended topology.
        parents = list(MANO_PARENT_21) + [0] * (num_joints - len(MANO_PARENT_21))
    return torch.tensor(parents, dtype=torch.long, device=device)


class HandJointTransformerEncoder(nn.Module):
    """Encode MANO joints into spatially aligned hand-condition tokens.

    This is an opt-in articulation encoder inspired by skeleton-transformer
    practice (joint identity embeddings, temporal embeddings, local kinematic
    features, and self-attention over the joint-time token sequence). It returns
    one token per MANO joint per timestep, using the original world-space joint
    coordinate as the token coordinate so the downstream PointTransformer remains
    geometrically aligned with the scene/object points.
    """

    def __init__(self, args, channels: int):
        super().__init__()
        self.channels = int(channels)
        self.num_joints = int(getattr(args, "hand_encoder_num_joints", 21))
        self.mano_param_dim = int(getattr(args, "hand_encoder_mano_param_dim", 64))

        raw_dim = 14 + self.mano_param_dim  # xyz, wrist-local xyz, velocity, bone vector/length, valid
        self.input_proj = MLP(raw_dim, self.channels, self.channels)
        self.time_embed = TemporalEmbedding(self.channels)
        self.joint_embed = nn.Embedding(self.num_joints, self.channels)
        self.hand_side_embed = nn.Embedding(2, self.channels)
        self.valid_embed = nn.Embedding(2, self.channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channels,
            nhead=int(getattr(args, "hand_encoder_heads", 4)),
            dim_feedforward=self.channels * 4,
            dropout=float(getattr(args, "hand_encoder_dropout", 0.1)),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(getattr(args, "hand_encoder_layers", 4)),
            norm=nn.LayerNorm(self.channels),
        )

    def _collect_mano_params(self, data_dict: dict, B: int, T: int, H: int, device, dtype) -> torch.Tensor:
        if self.mano_param_dim == 0:
            return torch.zeros(B, T, H, 0, device=device, dtype=dtype)

        pieces: list[torch.Tensor] = []
        for key in MANO_PARAM_KEYS:
            value = data_dict.get(key)
            if not isinstance(value, torch.Tensor):
                continue
            value = value.to(device=device, dtype=dtype)
            if value.ndim < 3:
                continue
            if value.shape[0] != B:
                continue

            # Supported server-side export shapes:
            #   (B,T,2,D...), e.g. MANO PCA/full pose and wrist transforms per frame.
            #   (B,T,D...),   e.g. sequence MANO params shared/broadcast to both hands.
            #   (B,2,D...),   e.g. sequence-constant betas per hand.
            # Other shapes are ignored rather than guessed, preserving older datasets.
            if value.shape[1] == T and value.ndim >= 4 and value.shape[2] <= H:
                per_hand = value
            elif value.shape[1] == T and value.ndim == 3 and value.shape[2] <= H:
                per_hand = value.unsqueeze(-1)
            elif value.shape[1] == T:
                per_hand = value.unsqueeze(2).expand(-1, -1, H, *value.shape[2:])
            elif value.shape[1] == H:
                per_hand = value.unsqueeze(1).expand(-1, T, -1, *value.shape[2:])
            else:
                continue
            if per_hand.shape[2] < H:
                pad_shape = (B, T, H - per_hand.shape[2]) + tuple(per_hand.shape[3:])
                per_hand = torch.cat([per_hand, per_hand.new_zeros(pad_shape)], dim=2)
            per_hand = per_hand[:, :, :H].reshape(B, T, H, -1)
            pieces.append(per_hand)

        if pieces:
            params = torch.cat(pieces, dim=-1)
        else:
            params = torch.zeros(B, T, H, 0, device=device, dtype=dtype)
        if params.shape[-1] >= self.mano_param_dim:
            return params[..., : self.mano_param_dim]
        pad = params.new_zeros(B, T, H, self.mano_param_dim - params.shape[-1])
        return torch.cat([params, pad], dim=-1)

    def forward(self, data_dict: dict, target_T: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        joints = data_dict.get("hand_joints_world")
        if not isinstance(joints, torch.Tensor):
            raise KeyError(
                "--hand_condition_encoder=joint_transformer requires batch['hand_joints_world'] "
                "with shape (B,T,2,J,3). Reprocess or update the HOI NPZ loader to pass MANO joints."
            )
        if joints.ndim != 5 or joints.shape[2] < 1 or joints.shape[-1] != 3:
            raise ValueError(f"hand_joints_world must have shape (B,T,2,J,3), got {tuple(joints.shape)}")
        if joints.shape[1] != target_T:
            raise ValueError(
                f"hand_joints_world timestep horizon ({joints.shape[1]}) must match model horizon ({target_T})"
            )

        B, T, H, J_src, _ = joints.shape
        H = min(int(H), 2)
        J = min(int(J_src), self.num_joints)
        device = joints.device
        dtype = joints.dtype
        joints = joints[:, :, :H, :J].float()

        valid = data_dict.get("hand_joint_valid")
        if isinstance(valid, torch.Tensor) and valid.ndim == 4 and valid.shape[:3] == (B, T, H):
            valid = valid[:, :, :H, :J].to(device=device).float() > 0.5
        else:
            valid = torch.isfinite(joints).all(dim=-1) & (joints.abs().sum(dim=-1) > 1e-8)
        valid = valid & torch.isfinite(joints).all(dim=-1)
        joints = torch.nan_to_num(joints, nan=0.0, posinf=0.0, neginf=0.0)

        wrist = joints[:, :, :, :1]
        local = joints - wrist

        velocity = joints.new_zeros(joints.shape)
        velocity[:, 1:] = joints[:, 1:] - joints[:, :-1]

        parents = _mano_parent_indices(J, device=device)
        clamped_parents = parents.clamp(min=0)
        parent_xyz = joints[:, :, :, clamped_parents]
        bone_vec = torch.where((parents >= 0).view(1, 1, 1, J, 1), joints - parent_xyz, torch.zeros_like(joints))
        bone_len = torch.linalg.norm(bone_vec, dim=-1, keepdim=True)

        mano_params = self._collect_mano_params(data_dict, B, T, H, device, dtype=torch.float32)
        mano_params = mano_params[:, :, :, None, :].expand(-1, -1, -1, J, -1)

        raw = torch.cat(
            [
                joints,
                local,
                velocity,
                bone_vec,
                bone_len,
                valid.float().unsqueeze(-1),
                mano_params,
            ],
            dim=-1,
        )
        feat = self.input_proj(raw)

        joint_ids = torch.arange(J, device=device)
        hand_ids = torch.arange(H, device=device)
        joint_emb = self.joint_embed(joint_ids).view(1, 1, 1, J, self.channels)
        hand_emb = self.hand_side_embed(hand_ids).view(1, 1, H, 1, self.channels)
        time_steps = torch.linspace(0, 1, T, device=device, dtype=torch.float32)
        time_emb = self.time_embed(time_steps.view(1, T)).view(1, T, 1, 1, self.channels)
        valid_emb = self.valid_embed(valid.long())
        feat = feat + joint_emb + hand_emb + time_emb + valid_emb

        seq = feat.reshape(B, T * H * J, self.channels)
        pad_mask = ~valid.reshape(B, T * H * J)
        # PyTorch attention returns NaNs if a row masks every token. Keep those rows
        # unmasked during encoding, then mark them invalid again in the returned mask.
        all_masked = pad_mask.all(dim=1)
        if all_masked.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_masked] = False
        encoded = self.encoder(seq, src_key_padding_mask=pad_mask)
        encoded = encoded.reshape(B, T, H * J, self.channels)
        coords = joints.reshape(B, T, H * J, 3)
        exists = valid.reshape(B, T, H * J)
        return coords, encoded, exists


class ManoRelativeActionTokenEncoder(nn.Module):
    """Encode compact MANO control/action state as per-hand temporal tokens.

    This stream is intentionally separate from the joint-token encoder: MANO
    joints describe where the articulated geometry is, while this encoder carries
    wrist pose, relative motion, pose deltas, and shape/control features as a
    low-token action signal inspired by relative-action conditioning.
    """

    def __init__(self, args, channels: int):
        super().__init__()
        self.channels = int(channels)
        self.pose_dim = int(getattr(args, "hand_action_mano_pose_dim", 45))
        self.beta_dim = int(getattr(args, "hand_action_beta_dim", 10))
        raw_dim = 28 + 3 * self.pose_dim + self.beta_dim
        self.input_proj = MLP(raw_dim, self.channels, self.channels)
        self.time_embed = TemporalEmbedding(self.channels)
        self.hand_side_embed = nn.Embedding(2, self.channels)
        self.valid_embed = nn.Embedding(2, self.channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channels,
            nhead=int(getattr(args, "hand_action_encoder_heads", 4)),
            dim_feedforward=self.channels * 4,
            dropout=float(getattr(args, "hand_action_encoder_dropout", 0.1)),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(getattr(args, "hand_action_encoder_layers", 2)),
            norm=nn.LayerNorm(self.channels),
        )

    def _collect_per_hand(
        self,
        data_dict: dict,
        keys: tuple[str, ...],
        B: int,
        T: int,
        H: int,
        device,
        dtype,
    ) -> torch.Tensor | None:
        for key in keys:
            value = data_dict.get(key)
            if not isinstance(value, torch.Tensor):
                continue
            value = value.to(device=device, dtype=dtype)
            if value.ndim < 3 or value.shape[0] != B:
                continue
            if value.shape[1] == T and value.ndim >= 4 and value.shape[2] <= H:
                per_hand = value
            elif value.shape[1] == T:
                per_hand = value.unsqueeze(2).expand(-1, -1, H, *value.shape[2:])
            elif value.shape[1] == H:
                per_hand = value.unsqueeze(1).expand(-1, T, -1, *value.shape[2:])
            else:
                continue
            if per_hand.shape[2] < H:
                pad_shape = (B, T, H - per_hand.shape[2]) + tuple(per_hand.shape[3:])
                per_hand = torch.cat([per_hand, per_hand.new_zeros(pad_shape)], dim=2)
            return per_hand[:, :, :H].reshape(B, T, H, -1)
        return None

    def _pad_or_truncate(self, value: torch.Tensor | None, B: int, T: int, H: int, dim: int, device, dtype) -> torch.Tensor:
        if dim == 0:
            return torch.zeros(B, T, H, 0, device=device, dtype=dtype)
        if value is None:
            return torch.zeros(B, T, H, dim, device=device, dtype=dtype)
        if value.shape[-1] >= dim:
            return value[..., :dim]
        pad = value.new_zeros(B, T, H, dim - value.shape[-1])
        return torch.cat([value, pad], dim=-1)

    def _fallback_wrist_from_joints(self, data_dict: dict, B: int, T: int, H: int, device, dtype):
        joints = data_dict.get("hand_joints_world")
        if not isinstance(joints, torch.Tensor) or joints.ndim != 5 or joints.shape[-1] != 3:
            trans = torch.zeros(B, T, H, 3, device=device, dtype=dtype)
            valid = torch.zeros(B, T, H, device=device, dtype=torch.bool)
            return trans, valid
        joints = joints.to(device=device, dtype=dtype)[:, :, :H]
        if joints.shape[2] < H:
            pad = joints.new_zeros(B, T, H - joints.shape[2], joints.shape[3], 3)
            joints = torch.cat([joints, pad], dim=2)
        trans = torch.nan_to_num(joints[:, :, :H, 0], nan=0.0, posinf=0.0, neginf=0.0)
        valid = data_dict.get("hand_joint_valid")
        if isinstance(valid, torch.Tensor) and valid.ndim == 4 and valid.shape[:3] == (B, T, min(H, valid.shape[2])):
            if valid.shape[2] < H:
                pad = valid.new_zeros(B, T, H - valid.shape[2], valid.shape[3])
                valid = torch.cat([valid, pad], dim=2)
            valid = valid[:, :, :H, 0].to(device=device).float() > 0.5
        else:
            valid = torch.isfinite(trans).all(dim=-1) & (trans.abs().sum(dim=-1) > 1e-8)
        return trans, valid

    def _collect_wrist_pose(self, data_dict: dict, B: int, T: int, H: int, device, dtype):
        batch_shape = (B, T, H)
        rot = None
        trans = None

        wrist = self._collect_per_hand(
            data_dict,
            ("mano_wrist_transform", "mano_wrist_xform"),
            B, T, H, device, dtype,
        )
        if wrist is not None:
            if wrist.shape[-1] >= 16:
                mat = wrist[..., :16].reshape(B, T, H, 4, 4)
                rot = mat[..., :3, :3]
                trans = mat[..., :3, 3]
            elif wrist.shape[-1] >= 12:
                mat = wrist[..., :12].reshape(B, T, H, 3, 4)
                rot = mat[..., :3, :3]
                trans = mat[..., :3, 3]
            elif wrist.shape[-1] >= 6:
                rot = _axis_angle_to_matrix(wrist[..., :3])
                trans = wrist[..., 3:6]

        orient = self._collect_per_hand(data_dict, ("mano_global_orient",), B, T, H, device, dtype)
        translation = self._collect_per_hand(
            data_dict,
            ("mano_translation", "mano_transl", "mano_translation_xyz"),
            B, T, H, device, dtype,
        )
        if orient is not None and orient.shape[-1] >= 3:
            rot = _axis_angle_to_matrix(orient[..., :3])
        if translation is not None and translation.shape[-1] >= 3:
            trans = translation[..., :3]

        fallback_trans, fallback_valid = self._fallback_wrist_from_joints(data_dict, B, T, H, device, dtype)
        mano_valid = self._collect_per_hand(data_dict, ("mano_valid",), B, T, H, device, dtype)
        if rot is None:
            rot = _identity_rotation_matrix(batch_shape, device, dtype)
        if trans is None:
            trans = fallback_trans

        finite = torch.isfinite(rot).flatten(-2).all(dim=-1) & torch.isfinite(trans).all(dim=-1)
        valid = finite & (fallback_valid | (trans.abs().sum(dim=-1) > 1e-8))
        if mano_valid is not None and mano_valid.shape[-1] >= 1:
            valid = valid & (mano_valid[..., 0] > 0.5)
        rot = torch.nan_to_num(rot, nan=0.0, posinf=0.0, neginf=0.0)
        trans = torch.nan_to_num(trans, nan=0.0, posinf=0.0, neginf=0.0)
        return rot, trans, valid

    def forward(self, data_dict: dict, target_T: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scene = data_dict.get("scene_flows")
        if not isinstance(scene, torch.Tensor):
            raise KeyError("--hand_action_token_encoder requires batch['scene_flows'] to infer batch/device.")
        B = int(scene.shape[0])
        T = int(target_T)
        H = 2
        device = scene.device
        dtype = torch.float32

        rot, trans, valid = self._collect_wrist_pose(data_dict, B, T, H, device, dtype)
        pose = self._collect_per_hand(
            data_dict,
            ("mano_pose_coeffs", "mano_hand_pose", "mano_thetas"),
            B, T, H, device, dtype,
        )
        betas = self._collect_per_hand(data_dict, ("mano_betas",), B, T, H, device, dtype)
        pose = self._pad_or_truncate(pose, B, T, H, self.pose_dim, device, dtype)
        betas = self._pad_or_truncate(betas, B, T, H, self.beta_dim, device, dtype)

        rot0 = rot[:, :1]
        trans0 = trans[:, :1]
        rel_rot = torch.matmul(rot0.transpose(-1, -2), rot)
        rel_trans = torch.matmul(
            rot0.transpose(-1, -2),
            (trans - trans0).unsqueeze(-1),
        ).squeeze(-1)

        prev_rot = torch.cat([rot[:, :1], rot[:, :-1]], dim=1)
        prev_trans = torch.cat([trans[:, :1], trans[:, :-1]], dim=1)
        vel_rot = torch.matmul(prev_rot.transpose(-1, -2), rot)
        vel_rot[:, 0] = _identity_rotation_matrix((B, H), device, dtype)
        vel_trans = trans - prev_trans
        vel_trans[:, 0] = 0.0

        pose0 = pose[:, :1]
        pose_prev = torch.cat([pose[:, :1], pose[:, :-1]], dim=1)
        pose_rel = pose - pose0
        pose_vel = pose - pose_prev
        pose_vel[:, 0] = 0.0

        raw = torch.cat(
            [
                _rotation_6d_from_matrix(rot),
                trans,
                _rotation_6d_from_matrix(rel_rot),
                rel_trans,
                _rotation_6d_from_matrix(vel_rot),
                vel_trans,
                pose,
                pose_rel,
                pose_vel,
                betas,
                valid.float().unsqueeze(-1),
            ],
            dim=-1,
        )

        feat = self.input_proj(raw)
        hand_ids = torch.arange(H, device=device)
        hand_emb = self.hand_side_embed(hand_ids).view(1, 1, H, self.channels)
        time_steps = torch.linspace(0, 1, T, device=device, dtype=torch.float32)
        time_emb = self.time_embed(time_steps.view(1, T)).view(1, T, 1, self.channels)
        valid_emb = self.valid_embed(valid.long())
        feat = feat + hand_emb + time_emb + valid_emb

        seq = feat.permute(0, 2, 1, 3).reshape(B * H, T, self.channels)
        pad_mask = (~valid).permute(0, 2, 1).reshape(B * H, T)
        all_masked = pad_mask.all(dim=1)
        if all_masked.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_masked] = False
        encoded = self.encoder(seq, src_key_padding_mask=pad_mask)
        encoded = encoded.reshape(B, H, T, self.channels).permute(0, 2, 1, 3)
        return trans, encoded, valid


class BaseModel(nn.Module):
    def __init__(self, args, data_info_dict, rank: int = 0, cpu_pg=None):
        super().__init__()
        self.args = args
        self.device = args.device
        self.data_info_dict = data_info_dict
        self.T = _resolve_model_timesteps(args)
        self.context_horizon = CONTEXT_HORIZON
        self.pred_horizon = self.T - self.context_horizon
        self.rank = rank
        self.world_size = dist.get_world_size() if args.distributed else 1
        self.channels = args.predictor_dim
        self.cpu_pg = cpu_pg
        # ---------------------------- scene encoder ---------------------------- #
        self.scene_feature_encoder = SceneFeatureEncoder(
            args,
            self.channels,
            data_info_dict,
            rank,
            self.normalize_scene_features,
        )

        self.robot_proj = MLP(data_info_dict['robot_features_dim'], self.channels, self.channels)
        self.time_embed = TemporalEmbedding(self.channels)
        self.spatial_embed = SpatialEmbedding(self.channels)
        self.robot_type_emb = nn.Parameter(torch.zeros(1, self.channels))
        self.hand_condition_encoder_name = str(getattr(args, "hand_condition_encoder", "none"))
        self.hand_condition_fusion = str(getattr(args, "hand_condition_fusion", "append"))
        self.hand_action_token_encoder_name = str(getattr(args, "hand_action_token_encoder", "none"))
        if self.hand_condition_encoder_name == "joint_transformer":
            self.hand_joint_encoder = HandJointTransformerEncoder(args, self.channels)
            self.joint_token_type_emb = nn.Parameter(torch.zeros(1, self.channels))
            # Only needed when mesh and joint tokens are fused together.
            if self.hand_condition_fusion == "append":
                self.surface_token_type_emb = nn.Parameter(torch.zeros(1, self.channels))
            else:
                self.surface_token_type_emb = None
        elif self.hand_condition_encoder_name == "none":
            self.hand_joint_encoder = None
            self.surface_token_type_emb = None
            self.joint_token_type_emb = None
        else:
            raise ValueError(f"Unsupported hand_condition_encoder={self.hand_condition_encoder_name!r}")
        if self.hand_action_token_encoder_name == "mano_relative_transformer":
            self.hand_action_encoder = ManoRelativeActionTokenEncoder(args, self.channels)
            self.action_token_type_emb = nn.Parameter(torch.zeros(1, self.channels))
        elif self.hand_action_token_encoder_name == "none":
            self.hand_action_encoder = None
            self.action_token_type_emb = None
        else:
            raise ValueError(f"Unsupported hand_action_token_encoder={self.hand_action_token_encoder_name!r}")
        self.register_buffer("time_steps", torch.linspace(0, 1, self.T))
        with torch.no_grad():
            nn.init.kaiming_normal_(self.robot_type_emb)
            if self.surface_token_type_emb is not None:
                nn.init.kaiming_normal_(self.surface_token_type_emb)
            if self.joint_token_type_emb is not None:
                nn.init.kaiming_normal_(self.joint_token_type_emb)
            if self.action_token_type_emb is not None:
                nn.init.kaiming_normal_(self.action_token_type_emb)

        # ---------------------------- dynamics predictor ---------------------------- #
        self.dynamics_predictor = DynamicsPredictor(args, self.channels, self.T)

        # ---------------------------- normalization stats ---------------------------- #
        self._init_norm_stats()

        # Initialize consecutive NaN outputs counter
        self.consecutive_nan_outputs_count = 0

    def _init_norm_stats(self) -> None:
        if getattr(self.args, "identity_norm_stats", False):
            stats = norm_stats_utils.build_identity_norm_stats(
                self.args,
                self.device,
                self.T,
                robot_features_dim=self.data_info_dict["robot_features_dim"],
                scene_features_dim=self.data_info_dict["scene_features_dim"],
            )
        else:
            stats = norm_stats_utils.load_norm_stats_from_json(
                self.args,
                self.device,
                self.T,
                var_floor=VAR_FLOOR,
            )
        self._domains = list(stats.domains)
        self._domain_to_index = dict(stats.domain_to_index)
        self.register_buffer("norm_stats_per_step_mean", stats.per_step_mean)  # (D,T,3)
        self.register_buffer("norm_stats_per_step_var", stats.per_step_var)    # (D,T,3)
        self.register_buffer("robot_norm_mean", stats.robot_mean)              # (D, Fr)
        self.register_buffer("robot_norm_var", stats.robot_var)                # (D, Fr)
        self.register_buffer("scene_norm_mean", stats.scene_mean)              # (D, Ds)
        self.register_buffer("scene_norm_var", stats.scene_var)                # (D, Ds)

    def normalize(self, og):
        return norm_stats_utils.normalize_output(
            og,
            self._current_domain_indices,
            self.norm_stats_per_step_mean,
            self.norm_stats_per_step_var,
        )

    def unnormalize(self, normalized):
        return norm_stats_utils.unnormalize_output(
            normalized,
            self._current_domain_indices,
            self.norm_stats_per_step_mean,
            self.norm_stats_per_step_var,
        )

    def normalize_robot_features(self, robot_features):
        return norm_stats_utils.normalize_robot_features(
            robot_features,
            self._current_domain_indices,
            self.robot_norm_mean,
            self.robot_norm_var,
        )

    def normalize_scene_features(self, scene_features):
        return norm_stats_utils.normalize_scene_features(
            scene_features,
            self._current_domain_indices,
            self.scene_norm_mean,
            self.scene_norm_var,
        )

    def encode_scene_features(self, data_dict):
        B = data_dict["scene_flows"].shape[0]
        if not hasattr(self, '_current_domain_indices'):
            batch_domains = data_dict['__domain__']
            assert isinstance(batch_domains, (list, tuple)), "data_dict['__domain__'] must be a list of domain strings"
            assert len(batch_domains) == B, f"Number of domains ({len(batch_domains)}) must match batch size ({B})"
            self._current_domain_indices = torch.tensor(
                [self._domain_to_index[dom] for dom in batch_domains],
                device=self.device, dtype=torch.long
            )

        return self.scene_feature_encoder(data_dict)

    # ------------------------------------------------------------------ #
    def forward(self, data_dict, training=True, encoded_scene_feat0=None):
        for name, tensor in data_dict.items():
            if isinstance(tensor, torch.Tensor):
                assert torch.isfinite(tensor).all(), f"{name} has NaNs"
        # ------------------------- unpack data ---------------------------- #
        # unpack data
        scene_coord0 = data_dict["scene_flows"][:, 0]  # (B,Ns,3)
        scene_feat0  = data_dict["scene_features"][:, 0]  # (B,Ns,Ds)
        scene_exists0 = data_dict["scene_exists"][:, 0]  # (B,Ns)
        robot_coord_seq = data_dict["robot_flows"]  # (B,T,Nr,3)
        robot_feat_seq  = data_dict["robot_features"]  # (B,T,Nr,Fr)
        robot_exists = data_dict["robot_exists"]  # (B,T,Nr)
        # shapes
        B, Ns, _ = scene_coord0.shape
        _, T, Nr, _ = robot_coord_seq.shape
        if T != self.T:
            raise ValueError(
                f"Batch timestep horizon ({T}) does not match model horizon ({self.T}). "
                "For HOI NPZ training, set --hoi_target_horizon to match the processed sample length."
            )

        # ---------------------------- set batch domain indices ---------------------------- #
        batch_domains = data_dict['__domain__']
        assert isinstance(batch_domains, (list, tuple)), "data_dict['__domain__'] must be a list of domain strings"
        assert len(batch_domains) == B, f"Number of domains ({len(batch_domains)}) must match batch size ({B})"
        for dom in batch_domains:
            assert dom in self._domain_to_index, f"Unknown domain '{dom}' in batch."
        self._current_domain_indices = torch.tensor(
            [self._domain_to_index[dom] for dom in batch_domains],
            device=self.device, dtype=torch.long
        )

        # ---------------------------- scene encoder ---------------------------- #
        if encoded_scene_feat0 is not None:
            scene_feat0 = encoded_scene_feat0
        else:
            scene_feat0 = self.encode_scene_features(data_dict)
        with torch.autocast('cuda', enabled=False):
            scene_feat0 = scene_feat0 + self.spatial_embed(scene_coord0.float())

        # ---------------------------- robot / hand condition features ---------------------------- #
        from training.phystwin_controls import unlabelled_rows
        tracked_control_rows = unlabelled_rows(data_dict, B, scene_coord0.device)
        has_tracked_controls = bool(tracked_control_rows.any())
        use_mesh_condition = (
            self.hand_joint_encoder is None or self.hand_condition_fusion == "append"
            or has_tracked_controls
        )
        if use_mesh_condition:
            robot_feat_seq = self.normalize_robot_features(robot_feat_seq)
            with torch.autocast('cuda', enabled=False):  # TODO: see if this resolves nan issue
                robot_raw = self.robot_proj(robot_feat_seq)  # (B, T, Nr, C)
                time_emb = self.time_embed(self.time_steps.view(1, T)).unsqueeze(2)  # (1, T, 1, C)
                time_emb = time_emb.expand(B, T, Nr, -1)  # (B, T, Nr, C)
                robot_pos_emb = self.spatial_embed(robot_coord_seq.float())  # (B, T, Nr, C)
                robot_feat = robot_raw + robot_pos_emb + time_emb + self.robot_type_emb.view(1, 1, 1, -1)  # (B, T, Nr, C)

        if self.hand_joint_encoder is not None:
            if self.hand_condition_fusion == "append":
                assert self.surface_token_type_emb is not None
                robot_feat = robot_feat + self.surface_token_type_emb.view(1, 1, 1, -1)
            hand_coord_seq, hand_feat, hand_exists = self.hand_joint_encoder(data_dict, target_T=T)
            hand_exists = hand_exists & ~tracked_control_rows[:, None, None]
            with torch.autocast('cuda', enabled=False):
                hand_feat = (
                    hand_feat
                    + self.spatial_embed(hand_coord_seq.float())
                    + self.joint_token_type_emb.view(1, 1, 1, -1)
                )
            if self.hand_condition_fusion == "append":
                robot_coord_seq = torch.cat([robot_coord_seq, hand_coord_seq], dim=2)
                robot_feat = torch.cat([robot_feat, hand_feat], dim=2)
                robot_exists = torch.cat([robot_exists, hand_exists], dim=2)
            elif self.hand_condition_fusion == "replace":
                if has_tracked_controls:
                    # Keep pretrained joint modules and parameter shapes intact.
                    mesh_valid = robot_exists & tracked_control_rows[:, None, None]
                    joint_valid = hand_exists & ~tracked_control_rows[:, None, None]
                    robot_coord_seq = torch.cat([robot_coord_seq, hand_coord_seq], dim=2)
                    robot_feat = torch.cat([robot_feat, hand_feat], dim=2)
                    robot_exists = torch.cat([mesh_valid, joint_valid], dim=2)
                else:
                    robot_coord_seq = hand_coord_seq
                    robot_feat = hand_feat
                    robot_exists = hand_exists
            else:
                raise ValueError(f"Unsupported hand_condition_fusion={self.hand_condition_fusion!r}")

        if self.hand_action_encoder is not None:
            action_coord_seq, action_feat, action_exists = self.hand_action_encoder(data_dict, target_T=T)
            action_exists = action_exists & ~tracked_control_rows[:, None, None]
            with torch.autocast('cuda', enabled=False):
                action_feat = (
                    action_feat
                    + self.spatial_embed(action_coord_seq.float())
                    + self.action_token_type_emb.view(1, 1, 1, -1)
                )
            robot_coord_seq = torch.cat([robot_coord_seq, action_coord_seq], dim=2)
            robot_feat = torch.cat([robot_feat, action_feat], dim=2)
            robot_exists = torch.cat([robot_exists, action_exists], dim=2)

        out = self.dynamics_predictor(
            scene_coord0, scene_feat0, scene_exists0,
            robot_coord_seq, robot_feat, robot_exists,
            normalize_fn=self.normalize,
            unnormalize_fn=self.unnormalize,
            training=training
        )

        pred_norm = out["pred"]
        pred = self.unnormalize(pred_norm)  # (B,T,Ns,3)
        out["scene_relative_norm"] = pred_norm
        out["scene_relative"] = pred
        out["scene_flows"] = scene_coord0.unsqueeze(1) + pred  # (B,T,Ns,3)

        # Check outputs for NaN values with consecutive counting logic
        should_skip_batch, self.consecutive_nan_outputs_count = handle_nan_outputs(
            out,
            self.consecutive_nan_outputs_count,
            context=f"forward pass (batch size: {B})"
        )

        # In distributed training, synchronize the skip decision across all ranks
        # to prevent NCCL timeout due to ranks being out of sync
        if self.args.distributed:
            import torch.distributed as dist
            # Convert skip decision to tensor for all_reduce
            skip_tensor = torch.tensor([1 if should_skip_batch else 0],
                                     device=self.device, dtype=torch.int)
            # Use all_reduce with MAX to ensure if any rank wants to skip, all skip
            dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX, group=self.cpu_pg)
            should_skip_batch = skip_tensor.item() > 0

        # Mark whether this output should be skipped due to NaN
        out['_has_nan_outputs'] = should_skip_batch

        log_var_vis = out.get("log_var")
        if log_var_vis is None:
            raise RuntimeError("Model outputs missing log_var; cannot derive confidence.")
        log_var_vis = _prepare_log_var_for_confidence(log_var_vis, data_dict["__domain__"])
        var = torch.exp(log_var_vis)
        if var.shape[-1] == 1:
            var_scalar = var.squeeze(-1)
        else:
            var_scalar = var.mean(dim=-1)
        conf = 1.0 - (var_scalar - VAR_FLOOR) / max(1e-12, (VAR_CEILING - VAR_FLOOR))
        out["confidence"] = conf

        return out

    # ------------------------------------------------------------------ #
    #  Loss function                                                     #
    # ------------------------------------------------------------------ #
    def _make_nan_dict(self, template_keys):
        return metrics_utils.make_nan_dict(template_keys)

    @torch.no_grad()
    def _collect_metrics(
        self,
        per_point_loss,            # (B,T,NS)
        weights,                      # (B,T,NS)
        moved_mask, static_mask,      # (B,T,NS)
        pred_exists_supervised,       # (B,T,NS)
        l2,                           # (B,T,NS)
        log_var,                      # Tensor (B,T,NS,[1|3])
        pred_norm=None                # Optional tensor (B,T,NS,3)
    ):
        return metrics_utils.collect_metrics(
            self.args,
            per_point_loss,
            weights,
            moved_mask,
            static_mask,
            pred_exists_supervised,
            l2,
            log_var,
            pred_norm,
            var_floor=VAR_FLOOR,
            var_ceiling=VAR_CEILING,
        )

    def _compute_single_output_loss(self, output_norm, gt_target_norm, weights, pred_exists_supervised,
                              log_var):
        return losses_utils.compute_single_output_loss(
            self.args,
            output_norm,
            gt_target_norm,
            weights,
            pred_exists_supervised,
            log_var,
            var_floor=VAR_FLOOR,
            var_ceiling=VAR_CEILING,
            uncertainty_logvar_weight=UNCERTAINTY_LOGVAR_WEIGHT,
        )

    @torch.no_grad()
    def _compute_single_output_metrics(self, output_flows, gt_flows, per_point_loss,
                                    weights, moved, static, pred_exists_supervised,
                                    log_var, pred_norm=None):
        return losses_utils.compute_single_output_metrics(
            self.args,
            output_flows,
            gt_flows,
            per_point_loss,
            weights,
            moved,
            static,
            pred_exists_supervised,
            log_var,
            pred_norm,
            var_floor=VAR_FLOOR,
            var_ceiling=VAR_CEILING,
        )

    def loss_fn(self, outputs, data_dict, training=True):
        return losses_utils.loss_fn(
            self,
            outputs,
            data_dict,
            training,
            var_floor=VAR_FLOOR,
            var_ceiling=VAR_CEILING,
            sim_domain_keywords=SIM_DOMAIN_KEYWORDS,
            sim_var_const=SIM_VAR_CONST,
            uncertainty_logvar_weight=UNCERTAINTY_LOGVAR_WEIGHT,
        )

    def _internal_loss_fn(self, outputs, data_dict, training=True):
        return losses_utils.internal_loss_fn(
            self,
            outputs,
            data_dict,
            training,
            var_floor=VAR_FLOOR,
            var_ceiling=VAR_CEILING,
            sim_domain_keywords=SIM_DOMAIN_KEYWORDS,
            sim_var_const=SIM_VAR_CONST,
            uncertainty_logvar_weight=UNCERTAINTY_LOGVAR_WEIGHT,
        )

    @torch.no_grad()
    def _collect_per_domain_metrics(self,
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
                                    pred_norm):
        return metrics_utils.collect_per_domain_metrics(
            self.args,
            self.device,
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
            pred_norm,
            self._compute_single_output_metrics,
        )
