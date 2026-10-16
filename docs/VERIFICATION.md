# Release verification

On 15 October 2026, the actual release code and checkpoints were checked on a Linux server with NVIDIA A800 80 GB GPUs. These checks establish functioning inference, training and packaging; they are not a new benchmark campaign.

## End-to-end predicted 3D → video

The DeXplicit 3D model checkpoint produced a finite `[1,35,12000,3]` forecast from an initial-only real PhysTwin input and supplied actions. The initial state passed the consistency check. Its unchanged states 0–32 were projected through the calibrated camera and passed to the selected DeXplicit renderer high/low pair.

The complete command generated **33 frames at 832 × 480**, using 30 steps, seed 17, CFG 5, shift 5 and the native expert boundary at 875. The final MP4 was downloaded, matched against its receipt hash, decoded in full and visually inspected across all 33 frames. The scene and hand motion decode coherently, but the hand separates from the cloth. This test demonstrates the connected inference path, not physical accuracy or new benchmark performance.

The forecast, lossless RGB controls, coverage masks, decoded RGB and video are saved separately with a hash receipt. No source geometry or existing qualitative assets were edited.

## Control correctness

- All 33 real control frames and all 33 coverage masks are exactly equal to the original native RGB-cache rasterizer on the same forecast and calibration.
- Thirteen CPU checks cover calibrated pinhole/fisheye projection, source occlusion, predicted-motion sensitivity, future-scene/validity independence, initial-state mismatch rejection, and processed-camera transforms with and without legacy metadata.
- A real control export from the installed wheel exactly matches the checked source output.
- The pipeline reads only initial scene appearance and supplied actions/cameras. The standard renderer-training preparation predicts its conditions itself. The trainer rejects measured, unlabelled, altered and legacy conditioning bundles.

## Training

Actual forward/backward/optimizer updates were exercised for each training path:

| Path | Loss | Gradient norm | Finite changed tensors |
|---|---:|---:|---:|
| Geometry, previous unchanged entry point | -3.5605266094 | 87.289 | 1,427 |
| DeXplicit renderer high-noise expert | 0.01373891812 | 0.03588660434 | 160 |
| DeXplicit renderer low-noise expert | 0.03761795536 | 0.04525998980 | 160 |
| Optional compact renderer, previous unchanged entry point | 0.04919086769 | 0.041035477 | 337 |

The new DeXplicit renderer test generated predicted 3D controls before opening the real target clip, encoded both through the published preparation command, trained each expert in its recorded noise interval, saved Safetensors, and checked all saved tensors for finiteness, actual changes and exact inference schema compatibility. It used fresh AdamW state. The smoke-test updates are not substituted for the selected release weights.

The geometry objective contains log-likelihood terms and can be negative. One update proves a working training path; it does not establish convergence or reproduce the historical staged campaigns.

## Weight identity and downloads

- Geometry: all 1,451 exported tensors match the selected original state bit for bit, including two non-learned diagnostic NaN sentinel buffers; 425,063,158 learned parameters.
- DeXplicit renderer: unchanged high/low step-3264 files; each contains 160 BF16 LoRA tensors and 30,670,848 parameters.
- Optional compact renderer: all 337 trainable tensors match the selected state, with 29,691,104 parameters.
- All four uploaded Hugging Face files match their expected byte sizes and SHA-256 values. Model card, manifest, configuration and checksum text also match the checked local copies.
- The published downloader was run from an empty cache/output directory and downloaded and hash-verified all three default checkpoints.
- Both frozen 14B experts, the text encoder and VAE used in the GPU tests have the exact hashes of the pinned public upstream Hugging Face revision. The base-file manifest is included as a package resource.

## Packaging and retained compact checks

The final wheel was installed into a separate directory. All **11 public module entry points** passed `--help`, and the installed control builder reproduced the real control arrays exactly. All 119 source/resource files checked in the wheel match the release checkout; required configuration, checkpoint and adapter-schema JSON files are present.

The unchanged compact renderer previously produced 33 RGB frames and matched the original native engine's generated latents exactly in the same runtime (maximum difference 0). Its older archived RGB comparison differed by 1.342/255 mean absolute value and is not a bitwise cross-runtime claim.

## Scope

Compatible existing CUDA dependencies were used; this is not verification of a fresh dependency installation on every GPU. The complete benchmark was not rerun. The selected renderer's historical GT-3D-conditioned scores must not be reused as scores for this new forecast-conditioned pipeline. The causal variant and its timing claims are not included.

Machine-readable details: [verification.json](verification.json).

The installed wheel ran the complete predicted-3D pipeline and both predicted-conditioning renderer training updates. All three default canonical checkpoint filenames were downloaded from a new cache and hash-verified.
