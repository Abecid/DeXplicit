# Input and training data

The release operates on prepared, licensed research data. It does not reconstruct a metric scene, infer camera calibration or estimate hand actions from a single photograph. Obtain raw datasets and asset licenses from their authors; see LICENSING.md.

## Geometry preparation

`geometry prepare` reads processed per-sequence NPZ files through `dataset_components/hoi_dataset.py`. At minimum the processed format contains `scene_flows [T,N,3]` and `robot_flows [T,M,3]`. This checkpoint additionally requires pointwise appearance features, point validity, object identities/canonical coordinates and the hand annotations appropriate to the domain. Scene feature vectors consumed by the selected checkpoint have 778 channels, including the processed appearance/context features; they are not raw RGB.

Relevant fields include `scene_dino_features`, `scene_features`, `scene_exists`, `robot_features`, `robot_exists`, `scene_object_id`, `scene_canonical_xyz`, `hand_joints_world`, `hand_joint_valid`, and supported MANO coefficient/validity fields. The loader documents accepted per-frame/per-point broadcasting and sidecars. Preserve left/right and joint ordering. World coordinates are in metres. Do not reuse camera-space coordinates as world coordinates.

The exported inference bundle is Safetensors with `domains` JSON metadata. Inputs are selected from `dexplicit.geometry.INPUT_KEYS`. All scene input states repeat the initial state; reference future scene targets are excluded. Training uses processed sequences with targets through the native loader. Inference outputs are compressed NPZ; reading them does not require pickle.

## End-to-end camera and image inputs

`dexplicit.infer` uses the initial-only geometry bundle, one RGB image, and a camera NPZ. Every forecast/action state must describe the same world coordinate frame and temporal sampling as the camera. States 0–32 are used for video; states 33–34 remain in the saved geometry forecast. There is no temporal interpolation or resampling in the bridge.

Camera NPZ fields:

| Field | Shape | Meaning |
|---|---|---|
| `camera_model` | scalar string | `pinhole` or `FISHEYE624` |
| `world2cam` | `[33,4,4]` or `[4,4]` | Rigid world-to-camera transforms, translations in metres |
| `source_hw` | `[33,2]` or `[2]` | Height and width of the supplied source RGB grid |
| `intrinsics` | `[33,3,3]` or `[3,3]` | Pinhole K, required for pinhole cameras |
| `projection_params` | `[33,15]` or `[15]` | HOT3D FISHEYE624 calibration, required for fisheye |
| `rotate_clockwise_90` | scalar bool | Apply clockwise image/raster rotation before resize; default false |
| `pixel_transform` | `[33,3,3]` or `[3,3]` | Optional preprocessing provenance; already incorporated into K |
| `model_hw` | `[33,2]` or `[2]` | Optional legacy processed-pinhole marker: K is already in model pixels |

Pinhole K must refer directly to the supplied image grid. For processed pinhole sidecars containing `model_hw`, supply the image at that size. Legacy sidecars without that marker use `source_hw`. Intrinsics are already transformed; `pixel_transform` is provenance and is **not applied twice**. These images must already be oriented. For raw-camera inputs, use `source_hw` and raw calibration; the image and rendered controls receive the same rotation, aspect-preserving resize and center crop.

The bridge keeps initial-valid object point identities and the predictor's coordinates unchanged. It samples initial RGB with joint object/hand depth testing, then renders native square splats (object radius 2, hand radius 3; hand depth bias 0.0001 m). Coverage is saved for inspection. Unknown appearance stays empty; future visibility labels are not consumed. Coverage masks are diagnostic and are not passed as a different VACE masking task.

The geometry bundle supplies hand/controller trajectories directly. It does not infer future actions. Provide camera trajectories as explicit controls; ground-truth evaluation poses must not be described as camera predictions. The code requires calibrated metric inputs, not an arbitrary uncalibrated photograph.

## DeXplicit renderer (compact) tensor bundle

The public boundary is a prepared tensor interface. `pack_video` converts an NPZ with these exact keys to Safetensors and validates the main shapes. Preparing the VAE/text tensors requires the pinned upstream Wan backend; constructing calibrated controls remains the caller’s responsibility.

| Key | Shape | Meaning |
|---|---|---|
| `reference` | `[1,16,1,60,104]` | Initial RGB VAE latent, separate reference slot |
| `reactive` | `[1,16,9,60,104]` | Encoded 33-frame control stream |
| `inactive` | `[1,16,9,60,104]` | Native inactive VACE latent stream |
| `context`, `negative_context` | `[1,512,4096]` | Positive/negative Wan text embeddings |
| `hand.camera_xyz` | `[1,33,2,21,3]` | Joints in each frame’s camera coordinates, metres |
| `hand.reference_xyz` | `[1,33,2,21,3]` | Same joints in the fixed reference camera frame |
| `hand.valid` | `[1,33,2,21]` | Valid anatomical joints, boolean |
| `hand.uv` | `[1,33,2,21,2]` | Pixel centers in the 832×480 output image |
| `hand.in_view` | `[1,33,2,21]` | In-view mask, boolean; not an occlusion label |
| `hand.relative_camera` | `[1,33,3,4]` | Relative camera transform |
| `transport.point_features` | `[N,16]` | Initial-image descriptors for persistent points |
| `transport.point_indices` | `[33,60,104]` | Visible point index per target cell; −1 means unknown |
| `transport.source_visible` | `[N]` | Valid source descriptors, boolean |

Hands are ordered left then right. Joint ordering follows the native hand-conditioner contract, with wrist at joint 0. All 33 frames remain action controls. `video_modules.transport.build_transport` can construct the compact feature representation from initial VAE features plus caller-provided projected point coordinates, depth and validity. It resolves collisions by depth and keeps unknown/newly revealed appearance masked.

This interface does not change the origin of the caller’s geometry. Preserve that provenance in your experiment manifest. Future RGB targets must not be put in inference bundles.

For training, add `target_latents [1,16,10,60,104]`: the separate reference slot followed by the nine temporal target slots. Pass `--training` to the packer. The manifest is a JSON array of training-bundle paths, resolved relative to the manifest:

```json
["train/clip001.safetensors", "train/clip002.safetensors"]
```
