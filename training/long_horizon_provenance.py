"""Research provenance for the 35-frame long-horizon ablation campaign.

This file is intentionally blunt.  The stages in PR #16 are not all faithful
reimplementations of prior papers.  Some reuse an existing PointWorld mechanism,
some adapt an equation from a paper, and some are new PointWorld-specific
syntheses motivated by several works.  Treating all of them as "paper ports"
would be scientifically false.

No third-party source code is copied into these modules.  Official repositories
were inspected to understand architectures and reported hyperparameters; every
runtime implementation is written against PointWorld's supervised 3D point,
object-ID, canonical-coordinate, and planned-hand-trajectory interfaces.

Adaptation labels
-----------------
``existing_project_baseline``
    Existing PointWorld code; a paper may motivate the representation, but the
    stage is not an external implementation.
``official_code_inspired_adaptation``
    An official repository exists and was inspected, but the implementation was
    translated to PointWorld rather than copied or reproduced exactly.
``paper_spec_adaptation``
    The paper equation/representation is public, but executable training code was
    unavailable or not released when this audit was written.
``project_specific_synthesis``
    A new design assembled for this project.  Papers motivate pieces of the
    design, but there is no claim that the complete architecture/loss appeared in
    or was validated by a cited work.

Last audited: 2026-08-28.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MethodProvenance:
    stage: int
    name: str
    adaptation_level: str
    papers: tuple[str, ...]
    official_code: tuple[str, ...]
    code_status: str
    borrowed: tuple[str, ...]
    pointworld_changes: tuple[str, ...]
    custom_hyperparameters: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


METHOD_PROVENANCE: dict[int, MethodProvenance] = {
    0: MethodProvenance(
        stage=0,
        name="pose_control",
        adaptation_level="existing_project_baseline",
        papers=(
            "SPOT: SE(3) Pose Trajectory Diffusion for Object-Centric Manipulation (arXiv:2411.00965)",
        ),
        official_code=(
            "https://github.com/NVlabs/object_centric_diffusion",
        ),
        code_status="SPOT code released; this stage reuses PointWorld's pre-existing supervised SE(3) head instead of SPOT's diffusion policy.",
        borrowed=(
            "Object-level SE(3) trajectory as the low-dimensional coherent motion state.",
        ),
        pointworld_changes=(
            "No new architecture: pure pose_scene_flows is the canonical control.",
            "The complete future hand trajectory conditions a deterministic supervised predictor, not a pose-trajectory diffusion policy.",
        ),
        custom_hyperparameters=(),
    ),
    1: MethodProvenance(
        stage=1,
        name="local_residual",
        adaptation_level="project_specific_synthesis",
        papers=(
            "SPOT (arXiv:2411.00965) for object-level SE(3)",
            "ParticleFormer (arXiv:2506.23126) for jointly retaining global and local 3D dynamics",
        ),
        official_code=(
            "https://github.com/NVlabs/object_centric_diffusion",
        ),
        code_status="No official implementation matching SE(3) plus this mean-free dense canonical residual was found.",
        borrowed=(
            "Use coherent object motion as the global component.",
            "Retain local point-level capacity for deformable/flexible objects.",
        ),
        pointworld_changes=(
            "Predict deformation in the object's canonical frame, then transform it with the predicted SE(3).",
            "Subtract the per-object residual mean so translation cannot leak through the deformation branch.",
            "Zero-initialize the residual decoder so the run starts exactly at the pose control.",
        ),
        custom_hyperparameters=(
            "LONG_HORIZON_MAX_DEFORMATION_M=0.08 is a project pilot value, not a paper default.",
            "The LayerNorm-MLP decoder and tanh bound are PointWorld-specific.",
        ),
    ),
    2: MethodProvenance(
        stage=2,
        name="structure_edge",
        adaptation_level="project_specific_synthesis",
        papers=(
            "ParticleFormer: A 3D Point Cloud World Model for Multi-Object, Multi-Material Robotic Manipulation (arXiv:2506.23126)",
        ),
        official_code=(),
        code_status="The ParticleFormer project page/paper were available, but no official code repository was linked during the audit.",
        borrowed=(
            "Supervise both global and local 3D structure for rigid, deformable, and flexible materials.",
        ),
        pointworld_changes=(
            "This is NOT ParticleFormer's reported hybrid point-cloud reconstruction loss.",
            "PointWorld has persistent correspondences, so the ablation compares canonical-KNN future edge lengths directly to GT future edge lengths.",
            "Using future GT edge lengths permits real deformation instead of imposing frame-zero rigidity.",
        ),
        custom_hyperparameters=(
            "KNN k=6, 128 sampled points/object, SmoothL1 beta=0.005 m, and loss weight=2.0 are project pilot values.",
        ),
    ),
    3: MethodProvenance(
        stage=3,
        name="contact_aux",
        adaptation_level="paper_spec_adaptation",
        papers=(
            "JointHOI: Jointly Generating Contact Maps Enhances Hand-Object Interaction Generation (arXiv:2607.01768)",
        ),
        official_code=(),
        code_status="The JointHOI project page exposed a Code link, but it did not resolve to accessible implementation code during the audit.",
        borrowed=(
            "Represent contact as a continuous, temporally varying distance-to-hand field rather than a binary label.",
            "Keep separate left- and right-hand contact channels.",
        ),
        pointworld_changes=(
            "JointHOI jointly diffuses motion and contact; this stage adds a deterministic auxiliary head to PointWorld.",
            "Future GT geometry is used only to construct supervised distance targets, never as an inference input.",
        ),
        custom_hyperparameters=(
            "128 contact points/object, tau=0.02 m, near-contact multiplier=4, rank=32, and loss weight=0.1 are project pilot values.",
            "The normalized dot-product contact head and log-distance SmoothL1 objective are not claimed to be JointHOI's implementation.",
        ),
    ),
    4: MethodProvenance(
        stage=4,
        name="contact_routing",
        adaptation_level="project_specific_synthesis",
        papers=(
            "JointHOI (arXiv:2607.01768) for dynamic contact fields",
            "Latent Particle World Models (arXiv:2603.04553) for entity/particle-specific transition context instead of a single global action",
        ),
        official_code=(
            "https://github.com/taldatech/lpwm",
        ),
        code_status="LPWM code was inspected, but no prior implementation of this PointWorld contact gate was ported.",
        borrowed=(
            "Interaction should be routed to the object regions/particles affected by the action.",
        ),
        pointworld_changes=(
            "Convert predicted metric distance to a Gaussian influence field and route known hand displacement to points.",
            "Mean-center the correction so global translation remains the SE(3) head's responsibility.",
        ),
        custom_hyperparameters=(
            "Gaussian tau=0.02 m, route floor=0.15, linear 3D route head, and 0.08 m correction bound are project-specific.",
        ),
    ),
    5: MethodProvenance(
        stage=5,
        name="contact_flow",
        adaptation_level="paper_spec_adaptation",
        papers=(
            "ContactFlow: A Video Action Conditioning that Transfers Across Embodiments (arXiv:2607.26579)",
        ),
        official_code=(
            "https://github.com/rpl-bonn/contactflow",
        ),
        code_status="The official repository explicitly said training/inference code and checkpoints were under construction on 2026-08-28.",
        borrowed=(
            "The seven-value contact representation [x,y,z, delta-x,delta-y,delta-z, confidence].",
            "Focus the action representation on the active object-contact interface rather than full actor geometry.",
        ),
        pointworld_changes=(
            "ContactFlow writes sparse 7-channel image-space controls into VACE/ControlNet and trains a video latent flow-matching model.",
            "This stage instead forms object-frame contact tokens and decodes a bounded 3D PointWorld residual. It is not a reproduction of their video model.",
        ),
        custom_hyperparameters=(
            "The 7->64->3 MLP, confidence aggregation, metric gate, and correction bound are project-specific because upstream executable code was unavailable.",
        ),
    ),
    6: MethodProvenance(
        stage=6,
        name="latent_particles",
        adaptation_level="official_code_inspired_adaptation",
        papers=(
            "Latent Particle World Models: Self-supervised Object-centric Stochastic Dynamics Modeling (arXiv:2603.04553)",
        ),
        official_code=(
            "https://github.com/taldatech/lpwm",
        ),
        code_status="Official LPWM PyTorch code and configs were inspected; this implementation is deliberately not a literal DLPv3/LPWM port.",
        borrowed=(
            "Use multiple persistent local particles/parts instead of one global object token or independent dense point motion.",
            "Provide per-particle transition context rather than mean-pooling everything into one action.",
        ),
        pointworld_changes=(
            "PointWorld already supplies object IDs, canonical 3D points, and persistent correspondences, so it does not relearn 2D keypoints/masks/appearance particles.",
            "The current ablation is deterministic and omits LPWM's inverse posterior, policy prior, KL term, AdaLN stochastic latent actions, and autoregressive image decoder.",
            "Soft assignments decode low-rank canonical 3D deformation under the existing object SE(3).",
        ),
        custom_hyperparameters=(
            "LONG_HORIZON_PARTICLES=16 and temperature=0.5 are compute-conscious pilot values, not LPWM defaults.",
            "LPWM's public Sketchy-action config uses 64 prior particles, 45 encoded particles, 512 hidden width, six dynamics layers/eight heads, beta_dyn=0.2, and beta_dyn_rec=1.0.",
            "Balance=0.02, entropy=0.002, and compactness=0.1 are PointWorld-specific initial regularizer weights.",
        ),
    ),
    7: MethodProvenance(
        stage=7,
        name="object_jepa_physical",
        adaptation_level="official_code_inspired_adaptation",
        papers=(
            "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (arXiv:2506.09985)",
            "DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning (arXiv:2411.04983)",
        ),
        official_code=(
            "https://github.com/facebookresearch/vjepa2",
            "https://github.com/gaoyuezhou/dino_wm",
        ),
        code_status="Official V-JEPA 2 and DINO-WM repositories were inspected; the stage is a smaller supervised PointWorld auxiliary, not either official training pipeline.",
        borrowed=(
            "Predict future state in a compact latent/feature space instead of relying only on dense reconstruction.",
            "Use pretrained DINO semantics as persistent object information.",
            "Require the action-conditioned hidden state to retain physically decodable pose/contact/deformation information.",
        ),
        pointworld_changes=(
            "V-JEPA 2 uses a learned online encoder, masked predictor, frozen EMA target encoder, and momentum schedule.",
            "This experiment deliberately uses fixed seeded projections of supervised 3D targets and a hidden-only MLP to isolate one auxiliary-loss hypothesis.",
            "DINO-WM predicts DINO patch features; this stage pools the PointWorld pointwise-DINO features by known object ID.",
        ),
        custom_hyperparameters=(
            "Latent dim=128, loss weight=0.10, block weights, 64 target points, and 1->2 horizon weighting are project pilot values.",
        ),
    ),
    8: MethodProvenance(
        stage=8,
        name="contact_physics",
        adaptation_level="project_specific_synthesis",
        papers=(
            "JointHOI (arXiv:2607.01768) for consistency between generated contact and geometry-implied contact",
        ),
        official_code=(),
        code_status="No accessible official JointHOI implementation was available during the audit.",
        borrowed=(
            "Penalize disagreement between predicted contact state and contact implied by synthesized geometry.",
        ),
        pointworld_changes=(
            "JointHOI's Contact Inner Guidance changes diffusion sampling at inference; this stage applies a supervised training loss to deterministic PointWorld outputs.",
            "The persistent-contact velocity term is a project-specific approximation and is GT-gated to avoid forcing all contacts to stick.",
        ),
        custom_hyperparameters=(
            "5 mm contact threshold, 1 cm/frame stick-relative-velocity threshold, consistency weight=0.1, and stick weight=0.05 are project pilot values.",
        ),
    ),
}


def method_provenance(stage: int) -> MethodProvenance:
    try:
        return METHOD_PROVENANCE[int(stage)]
    except KeyError as exc:
        raise ValueError(f"Expected stage 0..8, got {stage}") from exc


def provenance_contract(stage: int) -> dict[str, object]:
    """JSON-serializable provenance saved beside every experimental run."""
    return method_provenance(stage).to_dict()
