# SPDX-License-Identifier: Apache-2.0
"""Portable loading, inference and data preparation for the DeXplicit 3D model."""
from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).resolve().parent / "resources" / "geometry.json"
INPUT_KEYS = {
    "scene_flows", "scene_features", "scene_exists", "robot_flows",
    "robot_features", "robot_exists", "action_is_unlabelled",
    "hand_joints_world", "hand_joint_valid", "point_object_id", "point_canonical_xyz",
    "mano_pose_coeffs", "mano_hand_pose", "mano_thetas", "mano_betas",
    "mano_wrist_xform", "mano_wrist_transform", "mano_global_orient",
    "mano_translation", "mano_transl", "mano_translation_xyz", "mano_rot6d", "mano_valid",
}


def config(path=CONFIG, device="cuda"):
    args = argparse.Namespace(**json.loads(Path(path).read_text()))
    args.device = torch.device(device)
    args.distributed = False
    args.disable_compile = True
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(weights, config_path=CONFIG, device="cuda", *, train=False):
    """Load only tensors; no original experiment pickle or private paths needed."""
    from training.dexplicit_3d_mixed_fast import install
    install()
    from training.dexplicit_3d_model import DeXplicit3DModel
    args = config(config_path, device)
    seed_everything(args.seed)
    tensors = load_file(str(weights), device="cpu")
    info = {
        "scene_features_dim": tensors["scene_feature_encoder.scene_raw_feat_proj.weight"].shape[1],
        "robot_features_dim": tensors["robot_proj.fc1.weight"].shape[1],
    }
    with tempfile.TemporaryDirectory(prefix="dexplicit-model-") as scratch:
        args.output_dir = scratch
        model = DeXplicit3DModel(args, info, rank=0, cpu_pg=None)
        model.load_state_dict(tensors, strict=True)
    for name in ("_long_horizon_zero_init_checked", "_deformation_graph_identity_checked", "_carrier_gauge_identity_checked"):
        if hasattr(model.dynamics_predictor, name):
            setattr(model.dynamics_predictor, name, True)
    model.mixed_representation_contract = {"arm": "FULL", "forward": "DeXplicit 3D model, 35 states"}
    model = model.to(device).train(train)
    if not train:
        model.requires_grad_(False)
    return model, args


def initial_inputs(batch):
    clean = {key: value for key, value in batch.items() if key in INPUT_KEYS}
    clean.update({key: value for key, value in batch.get("pointworld_object_context", {}).items() if key in INPUT_KEYS})
    clean["__domain__"] = batch["__domain__"]
    for key in ("scene_flows", "scene_features", "scene_exists"):
        clean[key] = clean[key][:, :1].expand_as(clean[key]).contiguous()
    if bool(torch.as_tensor(clean.get("action_is_unlabelled", False)).any()):
        clean["robot_features"] = clean["robot_features"][:, :1].expand_as(clean["robot_features"]).contiguous()
    return clean


def dataset(roots, args, *, deterministic=True):
    from training.dynamics_experiments import ExperimentOptions, install_dataset_patch
    options = ExperimentOptions(
        mode=args.dynamics_experiment, require_pointwise_dino=True,
        local_increment_weight=args.local_increment_loss_weight,
        horizon_position_weight=args.horizon_position_loss_weight,
        horizon_weight_start=args.horizon_weight_start, horizon_weight_end=args.horizon_weight_end,
        velocity_weight=args.velocity_loss_weight, acceleration_weight=args.acceleration_loss_weight,
        object_displacement_weight=args.object_displacement_loss_weight,
        local_isometry_weight=args.local_isometry_loss_weight,
        local_isometry_k=args.local_isometry_k, local_isometry_decay=args.local_isometry_decay,
    )
    install_dataset_patch(options)
    from dataset_components.hoi_dataset import HOIDatasetConfig, HandObjectInteractionDataset
    cfg = HOIDatasetConfig(max_scene_points=args.max_scene_points, max_robot_points=args.max_robot_points,
                          target_horizon=args.hoi_target_horizon, deterministic=deterministic,
                          require_dino_features=True, seed=args.seed, domain=args.hoi_domain,
                          mano_sidecar_root=getattr(args, "hoi_mano_sidecar_root", None))
    return HandObjectInteractionDataset([str(Path(p)) for p in roots], cfg)


def save_inputs(batch, path):
    clean = initial_inputs(batch)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous().clone() for k, v in clean.items() if isinstance(v, torch.Tensor)},
              str(path), metadata={"domains": json.dumps(clean["__domain__"]), "format": "dexplicit-geometry-input-v1"})


def load_inputs(path, device):
    values = load_file(str(path), device=str(device))
    extra = set(values) - INPUT_KEYS
    if extra:
        raise ValueError(f"Unexpected model input fields: {sorted(extra)}")
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
    values["__domain__"] = json.loads(meta.get("domains", '["hand_world"]'))
    for key in ("scene_flows", "scene_features", "scene_exists"):
        if key not in values:
            raise ValueError(f"Missing {key}")
        values[key] = values[key][:, :1].expand_as(values[key]).contiguous()
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Export one processed HOI sample as inference-only tensors")
    prepare.add_argument("--data-dir", action="append", required=True)
    prepare.add_argument("--sample", required=True, help="NPZ stem in the dataset")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--config", default=str(CONFIG))
    infer = sub.add_parser("infer", help="Forecast 35 metric point states")
    infer.add_argument("--weights", required=True)
    infer.add_argument("--input", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--config", default=str(CONFIG))
    infer.add_argument("--device", default="cuda")
    cli = parser.parse_args()
    torch.set_num_threads(4)
    if cli.command == "prepare":
        args = config(cli.config, "cpu")
        seed_everything(args.seed)
        ds = dataset(cli.data_dir, args)
        lookup = {Path(p).stem: i for i, p in enumerate(ds.samples)}
        if cli.sample not in lookup:
            raise ValueError(f"Sample not found: {cli.sample}")
        from dataset_components.hoi_dataset import hoi_collate_fn
        save_inputs(hoi_collate_fn([ds[lookup[cli.sample]]], args), cli.output)
        print(json.dumps({"input": cli.output, "sample": cli.sample}))
    else:
        model, args = load_model(cli.weights, cli.config, cli.device)
        inputs = load_inputs(cli.input, cli.device)
        with torch.inference_mode():
            result = model(inputs, training=False)
        output = {k: v.detach().cpu().numpy() for k, v in result.items() if isinstance(v, torch.Tensor)}
        if not np.isfinite(output["scene_flows"]).all():
            raise FloatingPointError("The forecast contains nonfinite values")
        if not np.allclose(output["scene_flows"][:, 0], inputs["scene_flows"][:, 0].cpu().numpy(), atol=1e-5):
            raise ValueError("The forecast failed its initial-state consistency check")
        Path(cli.output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cli.output, **output)
        print(json.dumps({"output": cli.output, "shape": list(output["scene_flows"].shape), "seed": args.seed}))


if __name__ == "__main__":
    main()
