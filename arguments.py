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

import copy
import os
import sys
import numpy as np
from argparse import ArgumentParser

LOCAL_DATASET_DIR = os.environ.get('LOCAL_DATASET_DIR', '/dataset')
DOMAIN_TO_DATA_DIR = {
    'behavior': f'{LOCAL_DATASET_DIR}/behavior/wds',
    'droid': f'{LOCAL_DATASET_DIR}/droid/wds',
}

def str_to_bool(value):
    """Convert string representations to boolean values."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise ValueError(f"Invalid boolean value: {value}")

def str_to_none(value):
    """Convert 'none' or 'None' string to Python None, leave other values unchanged."""
    if isinstance(value, str) and value.lower() == 'none':
        return None
    return value


def parse_optional_int(value, arg_name: str):
    """Parse an optional integer CLI value where None means unset."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        parsed = int(value)
    elif isinstance(value, str):
        if value.lower() == "none":
            return None
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{arg_name} must be an integer or 'none', got {value!r}") from exc
    else:
        raise ValueError(f"{arg_name} must be an integer or 'none', got type {type(value).__name__}")
    if parsed < 1:
        raise ValueError(f"{arg_name} must be >= 1 when specified, got {parsed}")
    return parsed


def _collect_explicit_cli_dests(parser: ArgumentParser, argv: list[str]) -> set[str]:
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest
    explicit_options = set()
    for token in argv:
        if not token.startswith("-"):
            continue
        explicit_options.add(token.split("=", 1)[0])
    return {option_to_dest[opt] for opt in explicit_options if opt in option_to_dest}

def parse_args(skip_command_line=False):
    parser = ArgumentParser()
    # general
    parser.add_argument('--seed', '-s', type=int, default=-1)
    parser.add_argument('--deterministic_data', '-det', type=str, default='false', help='Use deterministic data')
    parser.add_argument('--deterministic_train', type=str, default='false', help='Use deterministic data pipeline for training (disables resampling)')
    parser.add_argument('--deterministic_algorithms', type=str, default='false',
                        help='Force torch deterministic algorithms (may error if unsupported ops are used)')
    parser.add_argument('--log_dir', '-ld', type=str, default='train_logs')
    parser.add_argument('--output-dir', '--output_dir', dest='output_dir', type=str, default=None,
                        help='Compatibility alias for HOI trainer output dir; maps to log_dir/exp_name when set.')
    parser.add_argument('--exp_name', '-en', type=str, default=None, help='Experiment name; if not provided, will be auto-generated')
    parser.add_argument(
        '--unique_run_dir',
        type=str,
        default='false',
        help='Append a timestamp/run id to fresh training output directories. HOI mode defaults this to true.',
    )
    parser.add_argument(
        '--run_id',
        type=str,
        default=None,
        help='Optional explicit run id suffix used when --unique_run_dir=true.',
    )
    parser.add_argument('--distributed', '-ddp', type=str, default='false', help='Use DDP')
    parser.add_argument('--ddp', action='store_true',
                        help='Compatibility alias for --distributed=true.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch_size', '--batch-size', '-b', dest='batch_size', type=int, default=22)
    parser.add_argument('--num_epochs', '--max-epochs', '-ne', dest='num_epochs', type=int, default=200)
    parser.add_argument('--num_workers', '--num-workers', '-nw', dest='num_workers', type=int, default=16)
    parser.add_argument('--eval_num_workers', '--eval-num-workers', '-enw', dest='eval_num_workers', type=int, default=5)
    parser.add_argument('--eval_freq', '-ef', type=int, default=600, help='Evaluate every N batches (equivalent to seconds in previous version)')
    parser.add_argument('--save_freq', '-sf', type=int, default=1800, help='Save checkpoint every N batches (equivalent to seconds in previous version)')
    parser.add_argument(
        '--save_best_only',
        type=str,
        default='false',
        help='Only overwrite model-last.pt when the checkpoint metric improves. HOI mode defaults this to true.',
    )
    parser.add_argument(
        '--checkpoint_metric',
        type=str,
        default='test/l2_moved/mean',
        help='Metric key used by --save_best_only.',
    )
    parser.add_argument(
        '--checkpoint_metric_mode',
        choices=['min', 'max'],
        default='min',
        help='Whether lower or higher checkpoint_metric is better.',
    )
    parser.add_argument('--num_eval_batches', '-ntb', type=int, default=10)
    parser.add_argument('--model_path', type=str, required=False, help='Path to the model checkpoint to load')
    parser.add_argument('--allow_optimizer_reset', type=str, default='false',
                        help='Allow optimizer reset if checkpoint optimizer state is incompatible')
    parser.add_argument(
        '--reset_optimizer_on_resume',
        '--reset-optimizer-on-resume',
        type=str,
        default='false',
        help='When loading --model_path for training, skip checkpoint optimizer state and start a fresh optimizer.',
    )
    parser.add_argument('--grad_clip_max_norm', '--grad-clip-max-norm', '-gcmn', dest='grad_clip_max_norm', type=float, default=5.0)
    parser.add_argument('--data_dirs', '-dd', type=str, default=None, help='comma separated list of data directories')
    parser.add_argument(
        '--dataset_type',
        type=str,
        default='pointworld_wds',
        choices=['pointworld_wds', 'hoi_npz'],
        help='Dataset backend: original PointWorld WebDataset shards or hand-object-interaction NPZ samples.',
    )
    parser.add_argument(
        '--hoi_train_dirs',
        type=str,
        default=None,
        help='Comma-separated directories or .npz files for HOI NPZ training samples.',
    )
    parser.add_argument(
        '--train-dirs',
        dest='hoi_train_dirs_list',
        nargs='+',
        type=str,
        default=None,
        help='Compatibility alias for --hoi_train_dirs; accepts space-separated paths.',
    )
    parser.add_argument(
        '--hoi_val_dirs',
        type=str,
        default=None,
        help='Comma-separated directories or .npz files for HOI NPZ validation/test samples. Defaults to train dirs.',
    )
    parser.add_argument(
        '--val-dirs',
        dest='hoi_val_dirs_list',
        nargs='+',
        type=str,
        default=None,
        help='Compatibility alias for --hoi_val_dirs; accepts space-separated paths.',
    )
    parser.add_argument(
        '--hoi_domain',
        type=str,
        default='hand_world',
        help='Domain label used for HOI batches and per-domain metrics.',
    )
    parser.add_argument(
        '--hoi_target_horizon',
        type=int,
        default=11,
        help='Number of timesteps emitted by the HOI loader. Default is 1 context + 10 prediction frames.',
    )
    parser.add_argument(
        '--hoi_require_dino_features',
        type=str,
        default='true',
        help='Require scene_dino_features when HOI samples do not already contain materialized scene_features.',
    )
    parser.add_argument(
        '--hoi_mano_sidecar_root',
        '--hoi-mano-sidecar-root',
        type=str,
        default=None,
        help=(
            'Optional root containing lightweight HOT3D MANO sidecars, such as '
            'data/processed/hot3d_clips_mano_sidecar. When unset, HOI samples are '
            'loaded exactly as stored in the main NPZ files.'
        ),
    )
    parser.add_argument(
        '--dino_cache_warmup',
        type=str,
        default='true',
        help='Before training, prebuild the uncompressed DINO context-frame sidecars with a progress bar '
             '(one-time, sharded across ranks). Set false to build lazily during the first epoch instead.',
    )
    parser.add_argument(
        '--dino_cache_warmup_workers',
        type=int,
        default=8,
        help='Per-rank thread count used to prebuild DINO sidecars during warmup.',
    )
    parser.add_argument(
        '--use_precomputed_scene_features',
        type=str,
        default='false',
        help='Project scene_features directly instead of running the camera/DINO scene encoder.',
    )
    parser.add_argument(
        '--identity_norm_stats',
        type=str,
        default='false',
        help='Use zero-mean/unit-variance normalization buffers instead of loading norm_stats_path.',
    )
    parser.add_argument('--native-pointworld', dest='native_pointworld', action='store_true',
                        help='Compatibility no-op: root train.py always uses the native PointWorld BaseModel.')
    parser.add_argument('--max_train_steps', type=int, default=-1, help='Stop training after this many train steps (<=0 disables)')
    parser.add_argument(
        '--train_splits',
        type=str,
        default=None,
        help='Optional comma-separated split override for training dataloader (for example: train or test).',
    )
    # robot sampler
    parser.add_argument('--max_robot_points', '--max-robot-points', '-mrp', dest='max_robot_points', type=int, default=500)
    # augmentation (fixed defaults; only keep camera/point caps)
    parser.add_argument('--max_scene_points', '--max-scene-points', '-msp', dest='max_scene_points', type=int, default=12000)
    parser.add_argument(
        '--train_min_num_cameras',
        type=str,
        default='1',
        help="Minimum number of cameras to sample during training ('none' = auto, clipped by per-sample availability).",
    )
    parser.add_argument(
        '--train_max_num_cameras',
        type=str,
        default='3',
        help="Maximum number of cameras to sample during training ('none' = auto, clipped by per-sample availability).",
    )
    parser.add_argument(
        '--eval_min_num_cameras',
        type=int,
        default=2,
        help='Minimum number of cameras to sample during evaluation.',
    )
    parser.add_argument(
        '--eval_max_num_cameras',
        type=int,
        default=2,
        help='Maximum number of cameras to sample during evaluation.',
    )
    # visualization (train-time visualization removed in release)
    # data
    parser.add_argument('--robot_features', '-rfeat', type=str, default='robot_flows,robot_colors,robot_normals,gripper_open,robot_velocity,robot_acceleration')
    parser.add_argument('--scene_features', '-sfeat', type=str, default='scene_flows,scene_colors,scene_normals,gripper_open,dist2robot')
    parser.add_argument('--domains', '-dom', type=str, default='', required=False, help='comma separated list of domains')
    # optimizer
    parser.add_argument('--base_lr', '--learning-rate', dest='base_lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', '--weight-decay', dest='weight_decay', type=float, default=0.01)
    # dynamics
    parser.add_argument('--grid_size', '-gs', type=float, default=0.015)
    # normalization
    parser.add_argument('--norm_stats_path', type=str, default='stats/droid', help='Path to folder containing precomputed JSON files with normalization statistics')
    # scene encoder (release: fixed to DINOv3 ViT-L16 multi-layer 2D)
    # compile / performance controls
    parser.add_argument('--disable_compile', type=str, default='false', help='Disable torch.compile for inference-only paths')
    # robot + deployment options: selection handled via deploy/robots.py (ROBOT_TYPE)
    # ----- DINOv3 aggregation options (fixed) -----
    parser.add_argument('--depth_threshold', '-dt', type=float, default=0.003, help='Depth threshold for visibility mask')
    # predictor
    parser.add_argument('--ptv3_size', '-ptv3s', type=str, default='base', help='Size of the ptv3 backbone (small|base|large)')
    parser.add_argument('--ptv3_patch_size', '-ptv3ps', type=int, default=256, help='Patch size for ptv3 backbone')
    parser.add_argument('--predictor_dim', '-pd', type=int, default=256, help='Dimension of predictor')
    parser.add_argument(
        '--hand_condition_encoder',
        '--hand-condition-encoder',
        dest='hand_condition_encoder',
        choices=['none', 'joint_transformer'],
        default='none',
        help=(
            "Optional HOI hand-condition encoder. 'none' preserves the current robot/hand point "
            "conditioning path; 'joint_transformer' encodes MANO joints as articulation-aware tokens."
        ),
    )
    parser.add_argument(
        '--hand_condition_fusion',
        '--hand-condition-fusion',
        dest='hand_condition_fusion',
        choices=['append', 'replace'],
        default='append',
        help=(
            "How joint_transformer tokens are fused with robot_flows tokens. 'append' keeps the "
            "current conditioning tokens and adds encoded MANO joint tokens; 'replace' uses only "
            "the encoded MANO joint tokens."
        ),
    )
    parser.add_argument('--hand_encoder_layers', '--hand-encoder-layers', dest='hand_encoder_layers', type=int, default=4)
    parser.add_argument('--hand_encoder_heads', '--hand-encoder-heads', dest='hand_encoder_heads', type=int, default=4)
    parser.add_argument('--hand_encoder_dropout', '--hand-encoder-dropout', dest='hand_encoder_dropout', type=float, default=0.1)
    parser.add_argument(
        '--hand_encoder_num_joints',
        '--hand-encoder-num-joints',
        dest='hand_encoder_num_joints',
        type=int,
        default=21,
        help='Number of MANO joints per hand consumed by the optional hand joint encoder.',
    )
    parser.add_argument(
        '--hand_encoder_mano_param_dim',
        '--hand-encoder-mano-param-dim',
        dest='hand_encoder_mano_param_dim',
        type=int,
        default=64,
        help=(
            "Fixed per-hand MANO parameter feature budget for the optional hand encoder. If arrays "
            "such as mano_pose_coeffs/mano_wrist_xform are present they are padded/truncated to this "
            "size and concatenated to each joint token; otherwise zeros are used."
        ),
    )
    parser.add_argument(
        '--hand_action_token_encoder',
        '--hand-action-token-encoder',
        dest='hand_action_token_encoder',
        choices=['none', 'mano_relative_transformer'],
        default='none',
        help=(
            "Optional MANO action-token stream. 'mano_relative_transformer' encodes per-hand "
            "MANO pose/wrist motion as compact relative action tokens and appends them to the "
            "active hand-condition tokens."
        ),
    )
    parser.add_argument('--hand_action_encoder_layers', '--hand-action-encoder-layers', dest='hand_action_encoder_layers', type=int, default=2)
    parser.add_argument('--hand_action_encoder_heads', '--hand-action-encoder-heads', dest='hand_action_encoder_heads', type=int, default=4)
    parser.add_argument('--hand_action_encoder_dropout', '--hand-action-encoder-dropout', dest='hand_action_encoder_dropout', type=float, default=0.1)
    parser.add_argument(
        '--hand_action_mano_pose_dim',
        '--hand-action-mano-pose-dim',
        dest='hand_action_mano_pose_dim',
        type=int,
        default=45,
        help='Fixed MANO hand-pose/PCA feature budget used by the optional relative action-token encoder.',
    )
    parser.add_argument(
        '--hand_action_beta_dim',
        '--hand-action-beta-dim',
        dest='hand_action_beta_dim',
        type=int,
        default=10,
        help='Fixed per-hand MANO beta feature budget used by the optional relative action-token encoder.',
    )
    # loss
    parser.add_argument('--huber_delta', '-hdl', type=float, default=5.0, help='Delta for huber loss')
    parser.add_argument(
        '--min_motion_loss_weight',
        type=float,
        default=None,
        help='Optional floor applied to raw selector loss weights before normalization. '
             'Unset preserves the original selector weighting exactly.',
    )
    parser.add_argument(
        '--balanced_moved_static_loss',
        type=str,
        default='false',
        help='Use balanced moved/static regression loss instead of selector-weighted regression.',
    )
    parser.add_argument(
        '--moved_loss_weight',
        type=float,
        default=1.0,
        help='Weight for moved-point regression when --balanced_moved_static_loss=true.',
    )
    parser.add_argument(
        '--static_loss_weight',
        type=float,
        default=1.0,
        help='Weight for static-point regression when --balanced_moved_static_loss=true.',
    )
    parser.add_argument(
        '--static_displacement_loss_weight',
        type=float,
        default=0.0,
        help='Optional penalty on predicted relative displacement magnitude for GT-static points.',
    )
    parser.add_argument(
        '--static_displacement_tau',
        type=float,
        default=0.005,
        help='Relative-motion threshold in meters for selecting GT-static points in the static displacement penalty.',
    )
    parser.add_argument(
        '--use_move_prediction_head',
        type=str,
        default='false',
        help='Add an auxiliary scene-point move/static prediction head trained from cumulative GT motion.',
    )
    parser.add_argument(
        '--move_prediction_loss_weight',
        type=float,
        default=0.1,
        help='Weight for the auxiliary move/static prediction loss.',
    )
    parser.add_argument(
        '--move_prediction_tau',
        type=float,
        default=0.005,
        help='Cumulative relative-motion threshold in meters used for move/static labels.',
    )
    parser.add_argument(
        '--move_prediction_temp_scale',
        type=float,
        default=5.0,
        help='Temperature scale for soft cumulative move labels.',
    )
    # aleatoric uncertainty
    # confidence threshold for uncertainty-aware metrics / viz
    parser.add_argument('--confidence_thres', '-cth', type=float, default=0.8,
                        help='Confidence threshold (0..1) for uncertainty-aware L2 metrics and filtering')
    # eval
    parser.add_argument('--eval_exp_name', '-ev', type=str, default=None)
    parser.add_argument('--eval_num_batches', '-enb', type=int, default=-1, help='Number of batches to evaluate (-1 for all)')
    parser.add_argument('--run_confidence_annotation', type=str, default='false',
                        help='Run a pass over eval_splits to store expert confidence arrays (B,T,Ns) per sample in H5 files; used later to compute filtered metrics.')
    parser.add_argument('--allow_missing_confidence_mask', type=str, default='false',
                        help='Allow evaluation to proceed if expert confidence masks are missing (skips filtered metrics).')
    parser.add_argument('--eval_viz_num', type=int, default=-1,
                        help='Number of eval samples to visualize (-1 disables visualization).')
    parser.add_argument('--eval_skip_viz', type=str, default='false',
                        help='Disable evaluation visualization even if eval_viz_num > 0.')
    parser.add_argument('--viewer_port', type=int, default=8080,
                        help='Viser viewer port for evaluation visualization.')
    parser.add_argument('--wandb', action='store_true', help='Enable online Weights & Biases logging.')
    parser.add_argument('--wandb-project', '--wandb_project', dest='wandb_project', type=str, default='point-world')
    parser.add_argument('--wandb-run-name', '--wandb_run_name', dest='wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb-mode', '--wandb_mode', dest='wandb_mode', type=str, default='online')
    if skip_command_line:
        raw_argv = []
        args = parser.parse_args([])
    else:
        raw_argv = sys.argv[1:]
        args = parser.parse_args()

    # make a deep copy of the args, this will be used for wandb sweeps
    args.og_args = copy.deepcopy(args)
    args._explicit_cli_dests = _collect_explicit_cli_dests(parser, raw_argv)

    # convert string arguments (which are supposed to be booleans or None) to bool or None
    for action in parser._actions:
        dest = action.dest
        if dest == 'help':
            continue
        val  = getattr(args, dest)
        # only care about str‑typed args
        if action.type is str:
            default = action.default
            if isinstance(default, str) and default.lower() in ('true','false'):
                setattr(args, dest, str_to_bool(val))
            elif default is None or (isinstance(default, str) and default.lower() == 'none'):
                setattr(args, dest, str_to_none(val))

    # Release-fixed defaults for trimmed CLI
    args.amp = True
    if args.ddp or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        args.distributed = True
    if args.output_dir is not None:
        output_dir = os.path.normpath(args.output_dir)
        parent, name = os.path.split(output_dir)
        explicit = set(getattr(args, "_explicit_cli_dests", set()))
        if "log_dir" not in explicit:
            args.log_dir = parent if parent else "."
        if "exp_name" not in explicit and name:
            args.exp_name = name
    if args.dataset_type == 'pointworld_wds' and (args.hoi_train_dirs is not None or args.hoi_train_dirs_list is not None):
        args.dataset_type = 'hoi_npz'

    if args.deterministic_train:
        args.deterministic_data = True
    args.train_min_num_cameras = parse_optional_int(args.train_min_num_cameras, "--train_min_num_cameras")
    args.train_max_num_cameras = parse_optional_int(args.train_max_num_cameras, "--train_max_num_cameras")
    if args.train_min_num_cameras is not None and args.train_max_num_cameras is not None:
        if args.train_min_num_cameras > args.train_max_num_cameras:
            raise ValueError(
                "--train_min_num_cameras must be <= --train_max_num_cameras "
                f"(got {args.train_min_num_cameras} > {args.train_max_num_cameras})"
            )
    if args.eval_min_num_cameras < 1:
        raise ValueError(f"--eval_min_num_cameras must be >= 1, got {args.eval_min_num_cameras}")
    if args.eval_max_num_cameras < 1:
        raise ValueError(f"--eval_max_num_cameras must be >= 1, got {args.eval_max_num_cameras}")
    if args.eval_min_num_cameras > args.eval_max_num_cameras:
        raise ValueError(
            "--eval_min_num_cameras must be <= --eval_max_num_cameras "
            f"(got {args.eval_min_num_cameras} > {args.eval_max_num_cameras})"
        )
    if args.min_motion_loss_weight is not None:
        args.min_motion_loss_weight = float(args.min_motion_loss_weight)
        if not (0.0 <= args.min_motion_loss_weight <= 1.0):
            raise ValueError(
                "--min_motion_loss_weight must be between 0 and 1 when set, "
                f"got {args.min_motion_loss_weight}"
            )
    # convert comma separated strings to lists
    args.robot_features = [name.strip() for name in args.robot_features.split(',')]
    args.scene_features = [name.strip() for name in args.scene_features.split(',')]

    expected_robot_features = [
        'robot_flows', 'robot_colors', 'robot_normals',
        'gripper_open', 'robot_velocity', 'robot_acceleration',
    ]
    expected_scene_features = [
        'scene_flows', 'scene_colors', 'scene_normals',
        'gripper_open', 'dist2robot',
    ]
    if args.robot_features != expected_robot_features:
        raise ValueError(
            f"Unsupported robot_features for release: {args.robot_features}. "
            f"Expected: {expected_robot_features}"
        )
    if args.scene_features != expected_scene_features:
        raise ValueError(
            f"Unsupported scene_features for release: {args.scene_features}. "
            f"Expected: {expected_scene_features}"
        )
    # validate and default
    args.domains = [name.strip() for name in args.domains.split(',')] if args.domains else []
    if args.dataset_type == 'hoi_npz':
        if not args.domains:
            args.domains = [args.hoi_domain]
        elif args.hoi_domain not in args.domains:
            args.hoi_domain = args.domains[0]

        def _parse_dirs(raw, arg_name):
            if raw is None:
                return None
            dirs = [item.strip() for item in raw.split(',') if item.strip()]
            if not dirs:
                raise ValueError(f"{arg_name} must include at least one path when specified")
            return dirs

        args.hoi_train_dirs = _parse_dirs(args.hoi_train_dirs, "--hoi_train_dirs")
        args.hoi_val_dirs = _parse_dirs(args.hoi_val_dirs, "--hoi_val_dirs")
        explicit = set(getattr(args, "_explicit_cli_dests", set()))
        if args.hoi_train_dirs_list is not None:
            args.hoi_train_dirs = args.hoi_train_dirs_list
        if args.hoi_val_dirs_list is not None:
            args.hoi_val_dirs = args.hoi_val_dirs_list
        if args.hoi_train_dirs is None and args.data_dirs is not None:
            args.hoi_train_dirs = [item.strip() for item in args.data_dirs.split(',') if item.strip()]
        if args.hoi_train_dirs is None:
            raise ValueError("--dataset_type=hoi_npz requires --hoi_train_dirs or --data_dirs")
        if args.hoi_val_dirs is None:
            args.hoi_val_dirs = args.hoi_train_dirs
        if args.hoi_target_horizon < 2:
            raise ValueError(
                f"--hoi_target_horizon must be at least 2 so there is one context frame and one prediction frame, "
                f"got {args.hoi_target_horizon}"
            )
        if "use_precomputed_scene_features" not in explicit:
            args.use_precomputed_scene_features = True
        if (not args.use_precomputed_scene_features) and "hoi_require_dino_features" not in explicit:
            args.hoi_require_dino_features = False
        if "identity_norm_stats" not in explicit:
            args.identity_norm_stats = True
        if "save_best_only" not in explicit:
            args.save_best_only = True
        if "unique_run_dir" not in explicit:
            args.unique_run_dir = True
    if any(domain.startswith('droid') for domain in args.domains):
        args.dynamics_head_init_scale = 1.0
    else:
        args.dynamics_head_init_scale = 0.0
    # map domains to data dirs if not provided
    if args.dataset_type == 'hoi_npz':
        args.data_dirs = args.hoi_train_dirs
    elif args.data_dirs is None:
        args.data_dirs = []
        for domain in args.domains:
            args.data_dirs.append(DOMAIN_TO_DATA_DIR[domain])
    else:
        args.data_dirs = [name.strip() for name in args.data_dirs.split(',')]
    if args.dataset_type == 'pointworld_wds':
        assert len(args.data_dirs) == len(args.domains), f'expected data_dirs and domains to have one to one mapping, got {len(args.data_dirs)} and {len(args.domains)}'
    # set a random random seed if None
    if args.seed == -1:
        import time
        args.seed = int(time.time_ns() + os.getpid()) % 1000000
    if args.eval_exp_name:
        args.amp = False  # often lead to NaN during eval

    if args.max_train_steps == 0:
        raise ValueError("--max_train_steps must be positive or -1")
    if args.max_train_steps < -1:
        raise ValueError("--max_train_steps must be -1 or a positive integer")
    if args.hand_condition_encoder != "none":
        if args.hand_encoder_layers < 1:
            raise ValueError(f"--hand_encoder_layers must be >= 1, got {args.hand_encoder_layers}")
        if args.hand_encoder_heads < 1:
            raise ValueError(f"--hand_encoder_heads must be >= 1, got {args.hand_encoder_heads}")
        if args.predictor_dim % args.hand_encoder_heads != 0:
            raise ValueError(
                "--hand_encoder_heads must divide --predictor_dim for the joint Transformer "
                f"(got heads={args.hand_encoder_heads}, predictor_dim={args.predictor_dim})"
            )
        if not (0.0 <= args.hand_encoder_dropout < 1.0):
            raise ValueError(f"--hand_encoder_dropout must be in [0, 1), got {args.hand_encoder_dropout}")
        if args.hand_encoder_num_joints < 1:
            raise ValueError(f"--hand_encoder_num_joints must be >= 1, got {args.hand_encoder_num_joints}")
        if args.hand_encoder_mano_param_dim < 0:
            raise ValueError(
                f"--hand_encoder_mano_param_dim must be >= 0, got {args.hand_encoder_mano_param_dim}"
            )
    if args.hand_action_token_encoder != "none":
        if args.hand_action_encoder_layers < 1:
            raise ValueError(f"--hand_action_encoder_layers must be >= 1, got {args.hand_action_encoder_layers}")
        if args.hand_action_encoder_heads < 1:
            raise ValueError(f"--hand_action_encoder_heads must be >= 1, got {args.hand_action_encoder_heads}")
        if args.predictor_dim % args.hand_action_encoder_heads != 0:
            raise ValueError(
                "--hand_action_encoder_heads must divide --predictor_dim for the MANO action Transformer "
                f"(got heads={args.hand_action_encoder_heads}, predictor_dim={args.predictor_dim})"
            )
        if not (0.0 <= args.hand_action_encoder_dropout < 1.0):
            raise ValueError(f"--hand_action_encoder_dropout must be in [0, 1), got {args.hand_action_encoder_dropout}")
        if args.hand_action_mano_pose_dim < 0:
            raise ValueError(f"--hand_action_mano_pose_dim must be >= 0, got {args.hand_action_mano_pose_dim}")
        if args.hand_action_beta_dim < 0:
            raise ValueError(f"--hand_action_beta_dim must be >= 0, got {args.hand_action_beta_dim}")

    if args.train_splits is not None:
        train_splits = [split.strip() for split in args.train_splits.split(',') if split.strip()]
        if len(train_splits) == 0:
            raise ValueError("--train_splits must include at least one split name")
        valid_splits = {"train", "test"}
        invalid_splits = [split for split in train_splits if split not in valid_splits]
        if invalid_splits:
            raise ValueError(
                f"--train_splits contains unsupported split(s): {invalid_splits}. "
                f"Supported splits: {sorted(valid_splits)}"
            )
        args.train_splits = train_splits[0] if len(train_splits) == 1 else train_splits

    return args
