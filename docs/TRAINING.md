# Training

## DeXplicit 3D model

`train_geometry` loads the DeXplicit 3D model with strict tensor matching and fine-tunes its native structured architecture and metric losses. The model forecasts future geometry from the initial scene and supplied actions. Recorded future geometry remains its supervised target; it is not substituted into renderer conditions.

Defaults are AdamW, learning rate 1e-4, configured weight decay, BF16 autocast, configured gradient clipping and batch size 1. The command saves model tensors, loss/gradient logs and local optimizer snapshots. `--wandb-project` enables online logging. Fine-tuning starts a fresh optimizer and does not reconstruct the original staged campaign.

## DeXplicit renderer: predicted 3D conditions

The standard training path and the inference path both call the same frozen DeXplicit 3D model and calibrated projection code. Training preparation predicts each trajectory itself; it does not accept precomputed control bundles or measured future object trajectories. It freezes the 3D model and keeps each forecast unchanged. Only the initial image supplies appearance; future hand and camera trajectories are supplied controls.

Prepare an initial-only geometry input with `dexplicit.geometry prepare`, then provide a JSON list with paths relative to the manifest:

```json
[
  {
    "source_id": "train-sequence-001",
    "split": "train",
    "geometry_input": "sequence-001/initial-and-actions.safetensors",
    "initial_image": "sequence-001/initial.png",
    "camera": "sequence-001/camera.npz",
    "video": "sequence-001/target.mp4",
    "prompt": "realistic egocentric hand object interaction video"
  }
]
```

The target video must contain exactly 33 RGB frames at 832 × 480, aligned to the initial image, supplied actions and camera. Future RGB is supervision only. It is opened after the 3D forecasts have been generated and never enters the predictor or the projected conditioning controls.

```bash
python -m dexplicit.prepare_renderer --manifest data/train.json \
  --geometry-weights weights/dexplicit-3d-model.safetensors \
  --geometry-python .venv-geometry/bin/python \
  --output-dir data/encoded --backend third_party/DiffSynth-Studio
python -m dexplicit.train_renderer --phase high \
  --weights weights/dexplicit-renderer-high.safetensors \
  --manifest data/encoded/manifest.json --output-dir runs/high \
  --backend third_party/DiffSynth-Studio
python -m dexplicit.train_renderer --phase low \
  --weights weights/dexplicit-renderer-low.safetensors \
  --manifest data/encoded/manifest.json --output-dir runs/low \
  --backend third_party/DiffSynth-Studio
```

Preparation saves each forecast, lossless projected controls and a receipt binding them to the 3D model checkpoint and initial-only inputs. The encoded training bundle references that receipt. Before allocating the renderer, training verifies predicted-conditioning metadata and the forecast/control checksums. Measured, unlabelled, altered or legacy bundles are rejected. Keep the `predicted-controls/` directory beside the encoded bundles when moving a training dataset.

Encoded tensors are `context [1,512,4096]`, `vace_context [1,96,10,60,104]` and `input_latents [1,16,10,60,104]`, including a separate initial-reference latent. The native Wan text, VACE and target VAE encoders are used. Training encoding is untiled; inference uses tiled encoding/decoding.

`train_renderer --phase high|low` loads the corresponding frozen backbone expert and trains rank-32 VACE LoRA at `q,k,v,o,ffn.0,ffn.2`. Its native flow-matching loss uses shift 5, BF16 model operations, FP32 trainable parameters, gradient checkpointing/offload, AdamW (learning rate 1e-4, weight decay 0.01) and gradient clipping at 1. The high and low noise intervals are fractions of the descending scheduler's index array: 0–0.358 and 0.358–1. The inference expert transition is raw timestep 875.

Saved weights use the DeXplicit renderer inference schema and record `conditioning_source=predicted_3d`. Pass them to `dexplicit.renderer`, or use the documented canonical weight filenames in a separate directory for `dexplicit.infer`. Keep the original evaluated checkpoints. Omit `--weights` to initialize fresh LoRA on the upstream expert.

## DeXplicit renderer: optional compact variant

The optional compact renderer retains its separate prepared-tensor interface, frozen Wan 1.3B backbone, rank-32 adapter and learned hand/feature modules. See DATA.md and `train_video`. Use `dexplicit.download --model renderer-compact` to download its weights. This optional variant is separate from the standard DeXplicit renderer training path above.

## Splits and reproducibility

Keep every window from a source sequence in one split. Use training sequences for fine-tuning, a fixed validation set for checkpoint selection and a separate test set for reported results. Record source identities, calibration, preprocessing versions and 3D model hashes. Curated examples and one-step release checks do not establish test-set performance.
