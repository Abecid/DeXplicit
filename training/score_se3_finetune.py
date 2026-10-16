"""Production staged fine-tuning for the score-aligned SE(3) campaign.

The campaign deliberately separates the *campaign step* (a synchronized
training batch consumed by every condition) from the *optimizer step* (an
actual AdamW update).  continued_baseline/metric_loss consume Phase A batches without updating any
parameter, while metric_pose--oracle_contact use those same batches to learn only the new adapter
and, for predicted_contact, the predicted-contact head.

This module owns three invariants which are easy to violate in an ad-hoc
trainer implementation:

* parameter groups are discovered by exact, audited names and are static for
  the lifetime of a run;
* every parameter outside the limited campaign groups is frozen before DDP is
  constructed; and
* disabled groups have ``grad=None`` before ``AdamW.step`` so they accumulate
  neither moments nor decoupled weight decay while their learning rate is
  zero.

Typical integration (all setup calls happen before DDP wrapping)::

    setup = prepare_score_se3_finetuning(model, candidate="predicted_contact", ...)
    trainer.optimizer = setup.optimizer
    trainer.lr_scheduler = setup.controller
    trainer.score_se3_optimizer_plan = setup.controller.optimizer_plan()

For each successful campaign batch, call ``controller.begin_batch(model)``
before forward/backward, then ``prepare_optimizer_step`` immediately after
backward and *before* ``GradScaler.unscale_``.  If it returns true, unscale and
clip only ``active_parameters`` before the normal scaler step/update.  If it
returns false, do not call unscale/step/update at all.  Finally call ``advance``
with whether AdamW actually stepped (AMP overflow means false).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn


SCORE_SE3_CANDIDATES = frozenset(
    {"continued_baseline", "metric_loss", "metric_pose", "predicted_contact", "oracle_contact", "dense_decoder", "deformation_graph", "gated_deformation_graph", "larger_deformation_graph"}
)
ADAPTER_CANDIDATES = frozenset({"metric_pose", "predicted_contact", "oracle_contact", "deformation_graph", "gated_deformation_graph", "larger_deformation_graph"})
PHASE_A_NOOP_CANDIDATES = frozenset({"continued_baseline", "metric_loss", "dense_decoder"})
DEFORMATION_GRAPH_CANDIDATES = frozenset({"dense_decoder", "deformation_graph", "gated_deformation_graph", "larger_deformation_graph"})
DEFORMATION_GRAPH_PREFIX = "dynamics_predictor.deformation_graph_head."

ADAPTER_PREFIX = "dynamics_predictor.score_se3_adapter."
CONTACT_PREFIXES = (
    "dynamics_predictor.long_horizon_hand_encoder.",
    "dynamics_predictor.long_horizon_contact_point.",
    "dynamics_predictor.long_horizon_contact_hand.",
)
CONTACT_EXACT_NAMES = (
    "dynamics_predictor.long_horizon_contact_bias",
)
BASE_POSE_PREFIX = "dynamics_predictor.pose."
FINAL_OBJECT_TIME_PREFIXES = (
    "dynamics_predictor.object_time.layers.6.",
    "dynamics_predictor.object_time.layers.7.",
)

ADAPTER_GROUP = "adapter_contact"
BASE_POSE_GROUP = "base_pose"
OBJECT_TIME_GROUP = "object_time_final_blocks"


def _validate_candidate(candidate: str) -> str:
    candidate = str(candidate)
    if candidate not in SCORE_SE3_CANDIDATES:
        raise ValueError(
            f"score-SE3 candidate must be one of {sorted(SCORE_SE3_CANDIDATES)}, "
            f"got {candidate!r}"
        )
    return candidate


def _finite_positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    value = float(value)
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not math.isfinite(value) or not lower_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}, got {value!r}")
    return value


@dataclass(frozen=True)
class ScoreSE3FineTuneConfig:
    """Immutable staged-optimization contract.

    The defaults are the frozen production schedule: 340 Phase-A campaign
    steps, followed by 510 Phase-B steps, for 850 total synchronized batches.
    """

    phase_a_steps: int = 340
    total_steps: int = 850
    warmup_steps: int = 20
    phase_a_adapter_lr: float = 1e-4
    phase_b_adapter_lr: float = 3e-5
    phase_b_base_lr: float = 3e-6
    weight_decay: float = 1e-2
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    schedule_mode: str = "staged_cosine"
    hold_steps: int = 0
    min_lr_ratio: float = 0.0

    def __post_init__(self) -> None:
        phase_a_steps = int(self.phase_a_steps)
        total_steps = int(self.total_steps)
        warmup_steps = int(self.warmup_steps)
        if self.schedule_mode not in {"staged_cosine", "warmup_hold_cosine"}:
            raise ValueError("unknown score-SE3 schedule mode")
        if self.schedule_mode == "warmup_hold_cosine":
            if not warmup_steps <= self.hold_steps < total_steps - 1:
                raise ValueError("continuation requires warmup_steps <= hold_steps < total_steps - 1")
            if not 0 < self.min_lr_ratio <= 1:
                raise ValueError("continuation min_lr_ratio must be in (0,1]")
        elif self.hold_steps != 0 or self.min_lr_ratio != 0:
            raise ValueError("hold/floor require warmup_hold_cosine")
        if phase_a_steps <= 0:
            raise ValueError("phase_a_steps must be positive")
        if total_steps <= phase_a_steps:
            raise ValueError("total_steps must be greater than phase_a_steps")
        if warmup_steps <= 0 or warmup_steps > phase_a_steps:
            raise ValueError(
                "warmup_steps must be positive and no greater than phase_a_steps"
            )
        _finite_positive(self.phase_a_adapter_lr, "phase_a_adapter_lr")
        _finite_positive(self.phase_b_adapter_lr, "phase_b_adapter_lr")
        _finite_positive(self.phase_b_base_lr, "phase_b_base_lr")
        _finite_positive(self.weight_decay, "weight_decay", allow_zero=True)
        _finite_positive(self.adam_eps, "adam_eps")
        if len(self.adam_betas) != 2:
            raise ValueError("adam_betas must contain exactly two values")
        for index, beta in enumerate(self.adam_betas):
            beta = float(beta)
            if not math.isfinite(beta) or not 0.0 <= beta < 1.0:
                raise ValueError(f"adam_betas[{index}] must be in [0,1)")

    @property
    def phase_b_steps(self) -> int:
        return int(self.total_steps) - int(self.phase_a_steps)

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["adam_betas"] = list(self.adam_betas)
        # Preserve the serialized contract of existing staged runs exactly.
        if self.schedule_mode == "staged_cosine":
            for key in ("schedule_mode", "hold_steps", "min_lr_ratio"):
                payload.pop(key)
        return payload

    @classmethod
    def from_args(cls, args: Any) -> "ScoreSE3FineTuneConfig":
        """Build the schedule from the campaign CLI namespace, fail closed.

        ``max_train_steps`` is a legacy trainer stop counter.  Full campaign
        launchers pin it to ``score_se3_total_steps``; isolated preflight
        smokes may use a smaller positive budget while exercising the exact
        same Phase-A schedule.  A value above the contracted total is never
        permitted.
        """

        config = cls(
            phase_a_steps=int(getattr(args, "score_se3_phase_a_steps", 340)),
            total_steps=int(getattr(args, "score_se3_total_steps", 850)),
            warmup_steps=int(getattr(args, "score_se3_warmup_steps", 20)),
            phase_a_adapter_lr=float(
                getattr(args, "score_se3_adapter_lr", 1e-4)
            ),
            phase_b_adapter_lr=float(
                getattr(
                    args,
                    "score_se3_adapter_phase_b_lr",
                    # Compatibility with the short-lived pre-launch spelling;
                    # production contracts record the canonical field above.
                    getattr(args, "score_se3_phase_b_adapter_lr", 3e-5),
                )
            ),
            phase_b_base_lr=float(getattr(args, "score_se3_base_lr", 3e-6)),
            weight_decay=float(getattr(args, "weight_decay", 1e-2)),
            schedule_mode=str(getattr(args, "score_se3_schedule_mode", "staged_cosine")),
            hold_steps=int(getattr(args, "score_se3_hold_steps", 0)),
            min_lr_ratio=float(getattr(args, "score_se3_min_lr_ratio", 0.0)),
        )
        max_train_steps = getattr(args, "max_train_steps", config.total_steps)
        if int(max_train_steps) <= 0 or int(max_train_steps) > int(
            config.total_steps
        ):
            raise ValueError(
                "score-SE3 requires 0 < max_train_steps <= "
                "score_se3_total_steps, "
                f"got {max_train_steps} vs {config.total_steps}"
            )
        return config


@dataclass(frozen=True)
class ScoreSE3ParameterGroupPlan:
    name: str
    parameter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter group name cannot be empty")
        if not self.parameter_names:
            raise ValueError(f"parameter group {self.name!r} cannot be empty")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError(f"parameter group {self.name!r} contains duplicate names")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_names": list(self.parameter_names),
        }


@dataclass(frozen=True)
class ScoreSE3ParameterPlan:
    """Exact trainable/frozen name partition established before DDP."""

    candidate: str
    groups: tuple[ScoreSE3ParameterGroupPlan, ...]
    frozen_parameter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_candidate(self.candidate)
        group_names = [group.name for group in self.groups]
        if len(set(group_names)) != len(group_names):
            raise ValueError("score-SE3 parameter group names must be unique")
        assigned = [name for group in self.groups for name in group.parameter_names]
        if len(set(assigned)) != len(assigned):
            raise ValueError("a parameter was assigned to more than one score-SE3 group")
        if set(assigned) & set(self.frozen_parameter_names):
            raise ValueError("trainable and frozen parameter names overlap")

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name for group in self.groups for name in group.parameter_names
        )

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "candidate": self.candidate,
            "groups": [group.as_dict() for group in self.groups],
            "trainable_parameter_names": list(self.trainable_parameter_names),
            "frozen_parameter_names": list(self.frozen_parameter_names),
        }


def _is_parallel_wrapper(model: nn.Module) -> bool:
    parallel_types = (nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    return isinstance(model, parallel_types)


def _matching_names(
    all_names: Sequence[str],
    *,
    prefixes: Iterable[str] = (),
    exact: Iterable[str] = (),
) -> tuple[str, ...]:
    prefix_tuple = tuple(prefixes)
    exact_set = frozenset(exact)
    return tuple(
        name
        for name in all_names
        if name in exact_set or any(name.startswith(prefix) for prefix in prefix_tuple)
    )


def build_score_se3_parameter_plan(
    model: nn.Module,
    candidate: str,
) -> ScoreSE3ParameterPlan:
    """Discover the exact static parameter topology and freeze everything else.

    This mutates ``requires_grad`` and therefore must be called before wrapping
    ``model`` in DDP.  Missing audited modules fail closed rather than silently
    broadening or shrinking the fine-tuning budget.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if _is_parallel_wrapper(model):
        raise RuntimeError(
            "build_score_se3_parameter_plan must run before DataParallel/DDP wrapping"
        )
    candidate = _validate_candidate(candidate)
    named_parameters = tuple(model.named_parameters())
    if not named_parameters:
        raise ValueError("model has no parameters")
    all_names = tuple(name for name, _ in named_parameters)

    base_prefixes = (BASE_POSE_PREFIX,)
    if candidate in DEFORMATION_GRAPH_CANDIDATES:
        # dense_decoder-larger_deformation_graph start from the trained metric_pose artifact.  Its residual pose adapter
        # is inherited machinery and co-adapts only at the low Phase-B rate.
        base_prefixes = (BASE_POSE_PREFIX, ADAPTER_PREFIX)
    base_pose = _matching_names(all_names, prefixes=base_prefixes)
    if not base_pose:
        raise RuntimeError(
            f"score-SE3 fine-tuning requires parameters under {BASE_POSE_PREFIX!r}"
        )

    final_blocks: list[str] = []
    for prefix in FINAL_OBJECT_TIME_PREFIXES:
        matched = _matching_names(all_names, prefixes=(prefix,))
        if not matched:
            raise RuntimeError(
                "score-SE3 fine-tuning requires exact final object-time block "
                f"parameters under {prefix!r}"
            )
        final_blocks.extend(matched)

    groups: list[ScoreSE3ParameterGroupPlan] = []
    if candidate in DEFORMATION_GRAPH_CANDIDATES - {"dense_decoder"}:
        graph = _matching_names(
            all_names, prefixes=(DEFORMATION_GRAPH_PREFIX,)
        )
        if not graph:
            raise RuntimeError(
                f"{candidate} requires graph parameters under "
                f"{DEFORMATION_GRAPH_PREFIX!r}"
            )
        groups.append(
            ScoreSE3ParameterGroupPlan(ADAPTER_GROUP, tuple(sorted(graph)))
        )
    elif candidate in ADAPTER_CANDIDATES:
        adapter = _matching_names(all_names, prefixes=(ADAPTER_PREFIX,))
        if not adapter:
            raise RuntimeError(
                f"{candidate} requires residual adapter parameters under "
                f"{ADAPTER_PREFIX!r}"
            )
        adapter_contact = list(adapter)
        if candidate == "predicted_contact":
            contact_names = _matching_names(
                all_names,
                prefixes=CONTACT_PREFIXES,
                exact=CONTACT_EXACT_NAMES,
            )
            for expected_prefix in CONTACT_PREFIXES:
                if not any(name.startswith(expected_prefix) for name in contact_names):
                    raise RuntimeError(
                        "predicted_contact requires predicted-contact parameters under "
                        f"{expected_prefix!r}"
                    )
            for exact_name in CONTACT_EXACT_NAMES:
                if exact_name not in contact_names:
                    raise RuntimeError(
                        f"predicted_contact requires predicted-contact parameter {exact_name!r}"
                    )
            adapter_contact.extend(contact_names)
        groups.append(
            ScoreSE3ParameterGroupPlan(
                ADAPTER_GROUP, tuple(sorted(adapter_contact))
            )
        )

    groups.extend(
        (
            ScoreSE3ParameterGroupPlan(BASE_POSE_GROUP, tuple(sorted(base_pose))),
            ScoreSE3ParameterGroupPlan(
                OBJECT_TIME_GROUP, tuple(sorted(final_blocks))
            ),
        )
    )
    trainable = {
        name for group in groups for name in group.parameter_names
    }
    frozen = tuple(sorted(set(all_names) - trainable))

    for name, parameter in named_parameters:
        parameter.requires_grad_(name in trainable)

    # Verify the mutation, including parameters that entered with a stale
    # requires_grad value from a previous experiment in this process.
    observed_trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if observed_trainable != trainable:
        raise RuntimeError(
            "failed to establish the exact score-SE3 trainable set: "
            f"missing={sorted(trainable - observed_trainable)}, "
            f"extra={sorted(observed_trainable - trainable)}"
        )
    return ScoreSE3ParameterPlan(candidate, tuple(groups), frozen)


def _parameters_for_group(
    model: nn.Module,
    group: ScoreSE3ParameterGroupPlan,
) -> list[nn.Parameter]:
    by_name = dict(model.named_parameters())
    missing = [name for name in group.parameter_names if name not in by_name]
    if missing:
        raise RuntimeError(
            f"model no longer matches parameter plan {group.name!r}; missing {missing}"
        )
    return [by_name[name] for name in group.parameter_names]


def build_score_se3_optimizer(
    model: nn.Module,
    plan: ScoreSE3ParameterPlan,
    config: ScoreSE3FineTuneConfig,
) -> torch.optim.AdamW:
    """Build AdamW once; its parameter topology never changes between phases."""

    if not isinstance(config, ScoreSE3FineTuneConfig):
        raise TypeError("config must be ScoreSE3FineTuneConfig")
    observed = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if observed != set(plan.trainable_parameter_names):
        raise RuntimeError(
            "model requires_grad state does not match the score-SE3 parameter plan"
        )

    parameter_groups: list[dict[str, Any]] = []
    for group in plan.groups:
        parameter_groups.append(
            {
                "params": _parameters_for_group(model, group),
                "name": group.name,
                "score_se3_parameter_names": list(group.parameter_names),
                # The controller installs the exact step-0 rates immediately.
                "lr": 0.0,
                "initial_lr": 0.0,
                "weight_decay": float(config.weight_decay),
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=0.0,
        betas=tuple(float(beta) for beta in config.adam_betas),
        eps=float(config.adam_eps),
        weight_decay=float(config.weight_decay),
    )
    return optimizer


class ScoreSE3FineTuneController:
    """Serializable campaign-step schedule and optimizer safety controller."""

    STATE_SCHEMA_VERSION = 1

    def __init__(
        self,
        candidate: str,
        parameter_plan: ScoreSE3ParameterPlan,
        config: ScoreSE3FineTuneConfig | None = None,
        *,
        campaign_step: int = 0,
        optimizer_step: int = 0,
    ) -> None:
        self.candidate = _validate_candidate(candidate)
        if parameter_plan.candidate != self.candidate:
            raise ValueError(
                "parameter plan candidate does not match controller candidate"
            )
        self.parameter_plan = parameter_plan
        self.config = config or ScoreSE3FineTuneConfig()
        self.campaign_step = self._validate_counter(
            campaign_step, "campaign_step", maximum=self.config.total_steps
        )
        self.optimizer_step = self._validate_counter(
            optimizer_step, "optimizer_step", maximum=self.campaign_step
        )
        self._validate_counter_relationship()
        self._last_lr = tuple(
            self.learning_rates(self.campaign_step).get(group.name, 0.0)
            for group in self.parameter_plan.groups
        )
        self._bound_optimizer_parameter_ids: tuple[tuple[int, ...], ...] | None = None

    @staticmethod
    def _validate_counter(value: int, name: str, *, maximum: int) -> int:
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{name} must be an integer")
        value = int(value)
        if value < 0 or value > int(maximum):
            raise ValueError(f"{name} must be in [0,{maximum}], got {value}")
        return value

    def _validate_counter_relationship(self) -> None:
        maximum_updates = self.campaign_step
        if self.candidate in PHASE_A_NOOP_CANDIDATES and self.config.schedule_mode == "staged_cosine":
            maximum_updates = max(
                0, self.campaign_step - int(self.config.phase_a_steps)
            )
        if self.optimizer_step > maximum_updates:
            raise ValueError(
                f"optimizer_step={self.optimizer_step} exceeds the maximum "
                f"{maximum_updates} updates possible by campaign_step="
                f"{self.campaign_step} for {self.candidate}"
            )

    @property
    def optimizer_phase(self) -> str:
        return self.phase_at(self.campaign_step)

    def phase_at(self, campaign_step: int) -> str:
        step = self._validate_counter(
            campaign_step, "campaign_step", maximum=self.config.total_steps
        )
        if self.config.schedule_mode == "staged_cosine" and step < int(self.config.phase_a_steps):
            return "phase_a"
        if step < int(self.config.total_steps):
            return "phase_b"
        return "complete"

    def should_optimizer_step(self, campaign_step: int | None = None) -> bool:
        raw_step = self.campaign_step if campaign_step is None else campaign_step
        phase = self.phase_at(raw_step)
        if phase == "complete":
            return False
        if phase == "phase_a" and self.candidate in PHASE_A_NOOP_CANDIDATES:
            return False
        return True

    def learning_rates(self, campaign_step: int | None = None) -> dict[str, float]:
        raw_step = self.campaign_step if campaign_step is None else campaign_step
        phase = self.phase_at(raw_step)
        step = int(raw_step)
        rates = {group.name: 0.0 for group in self.parameter_plan.groups}
        if phase == "complete":
            return rates

        if self.config.schedule_mode == "warmup_hold_cosine":
            if step < self.config.warmup_steps:
                factor = (step + 1) / self.config.warmup_steps
            elif step <= self.config.hold_steps:
                factor = 1.0
            else:
                # Reach the floor on the final *update*, then stop at completion.
                span = max(1, self.config.total_steps - 1 - self.config.hold_steps)
                progress = min(1.0, (step - self.config.hold_steps) / span)
                factor = self.config.min_lr_ratio + (1 - self.config.min_lr_ratio) * (
                    0.5 * (1 + math.cos(math.pi * progress)))
            if self.candidate in ADAPTER_CANDIDATES:
                rates[ADAPTER_GROUP] = self.config.phase_b_adapter_lr * factor
            rates[BASE_POSE_GROUP] = self.config.phase_b_base_lr * factor
            rates[OBJECT_TIME_GROUP] = self.config.phase_b_base_lr * factor
            return rates

        if phase == "phase_a":
            if self.candidate in ADAPTER_CANDIDATES:
                warmup_fraction = min(
                    1.0, (float(step) + 1.0) / float(self.config.warmup_steps)
                )
                rates[ADAPTER_GROUP] = (
                    float(self.config.phase_a_adapter_lr) * warmup_fraction
                )
            return rates

        progress = (
            float(step - int(self.config.phase_a_steps))
            / float(self.config.phase_b_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        if self.candidate in ADAPTER_CANDIDATES:
            rates[ADAPTER_GROUP] = float(self.config.phase_b_adapter_lr) * cosine
        rates[BASE_POSE_GROUP] = float(self.config.phase_b_base_lr) * cosine
        rates[OBJECT_TIME_GROUP] = float(self.config.phase_b_base_lr) * cosine
        return rates

    def _validate_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        if len(optimizer.param_groups) != len(self.parameter_plan.groups):
            raise RuntimeError(
                "optimizer parameter-group count does not match score-SE3 plan"
            )
        seen_parameter_ids: set[int] = set()
        ordered_parameter_ids: list[tuple[int, ...]] = []
        for observed, planned in zip(
            optimizer.param_groups, self.parameter_plan.groups
        ):
            observed_name = str(observed.get("name", ""))
            if observed_name != planned.name:
                raise RuntimeError(
                    "optimizer group order/name drift: expected "
                    f"{planned.name!r}, got {observed_name!r}"
                )
            recorded_names = tuple(observed.get("score_se3_parameter_names", ()))
            if recorded_names != planned.parameter_names:
                raise RuntimeError(
                    f"optimizer group {planned.name!r} parameter-name plan drift"
                )
            parameters = tuple(observed.get("params", ()))
            if len(parameters) != len(planned.parameter_names):
                raise RuntimeError(
                    f"optimizer group {planned.name!r} parameter count drift"
                )
            ids = {id(parameter) for parameter in parameters}
            ordered_parameter_ids.append(tuple(id(parameter) for parameter in parameters))
            if len(ids) != len(parameters) or ids & seen_parameter_ids:
                raise RuntimeError("optimizer parameter groups overlap or contain duplicates")
            seen_parameter_ids.update(ids)
            if float(observed.get("weight_decay", 0.0)) != float(
                self.config.weight_decay
            ):
                raise RuntimeError(
                    f"optimizer group {planned.name!r} weight_decay drift"
                )
        expected_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group.get("params", ())
        }
        if seen_parameter_ids != expected_ids:
            raise RuntimeError("optimizer topology validation failed")
        betas = tuple(float(value) for value in optimizer.defaults.get("betas", ()))
        if betas != tuple(float(value) for value in self.config.adam_betas):
            raise RuntimeError("optimizer Adam beta contract drift")
        if float(optimizer.defaults.get("eps", 0.0)) != float(self.config.adam_eps):
            raise RuntimeError("optimizer Adam epsilon contract drift")
        observed_binding = tuple(ordered_parameter_ids)
        if self._bound_optimizer_parameter_ids is None:
            self._bound_optimizer_parameter_ids = observed_binding
        elif observed_binding != self._bound_optimizer_parameter_ids:
            raise RuntimeError(
                "optimizer parameter-object topology changed after controller binding"
            )

    def apply_learning_rates(self, optimizer: torch.optim.Optimizer) -> dict[str, float]:
        self._validate_optimizer(optimizer)
        rates = self.learning_rates()
        for group in optimizer.param_groups:
            rate = float(rates[str(group["name"])])
            group["lr"] = rate
        self._last_lr = tuple(
            float(rates[group.name]) for group in self.parameter_plan.groups
        )
        return rates

    def begin_batch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """Clear every model gradient and install this campaign step's LRs."""

        if self.optimizer_phase == "complete":
            raise RuntimeError("score-SE3 campaign is already complete")
        model.zero_grad(set_to_none=True)
        return self.apply_learning_rates(optimizer)

    def prepare_optimizer_step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Discard disabled-group grads and report whether AdamW may run.

        Call this immediately after backward and before ``GradScaler.unscale_``.
        If it returns false, skip unscale, clipping, scaler step, and scaler
        update; calling unscale without a subsequent scaler update leaves
        GradScaler in an invalid per-optimizer state for the next batch.

        Clearing to ``None`` is essential: AdamW with LR=0 would still
        create/advance its moment state for a tensor with a non-None gradient.
        """

        rates = self.apply_learning_rates(optimizer)
        for group in optimizer.param_groups:
            if float(rates[str(group["name"])]) != 0.0:
                continue
            for parameter in group.get("params", ()):
                parameter.grad = None
        return self.should_optimizer_step()

    def active_parameters(
        self, optimizer: torch.optim.Optimizer
    ) -> tuple[nn.Parameter, ...]:
        """Return only currently enabled parameters for gradient clipping."""

        rates = self.apply_learning_rates(optimizer)
        return tuple(
            parameter
            for group in optimizer.param_groups
            if float(rates[str(group["name"])]) != 0.0
            for parameter in group.get("params", ())
        )

    def advance(
        self,
        *,
        optimizer_step_performed: bool,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Record one synchronized successful batch and optional AdamW update."""

        if not isinstance(optimizer_step_performed, bool):
            raise TypeError("optimizer_step_performed must be bool")
        if self.optimizer_phase == "complete":
            raise RuntimeError("cannot advance a completed score-SE3 campaign")
        if optimizer_step_performed and not self.should_optimizer_step():
            raise RuntimeError(
                f"AdamW must not step during {self.candidate} Phase A no-op"
            )
        self.campaign_step += 1
        if optimizer_step_performed:
            self.optimizer_step += 1
        self._validate_counter_relationship()
        if optimizer is not None:
            self.apply_learning_rates(optimizer)

    def get_last_lr(self) -> list[float]:
        """PyTorch-scheduler-compatible reporting surface."""

        return list(self._last_lr)

    def optimizer_plan(self) -> dict[str, Any]:
        """Return the complete human/machine-readable staged plan."""

        adapter_phase_a_updates = self.candidate in ADAPTER_CANDIDATES
        phase_a_rates = self.learning_rates(0)
        phase_b_rates = self.learning_rates(int(self.config.phase_a_steps))

        def planned_groups(
            rates: Mapping[str, float], *, schedule: str
        ) -> list[dict[str, Any]]:
            return [
                {
                    "name": group.name,
                    "parameter_names": list(group.parameter_names),
                    "lr": float(rates[group.name]),
                    "weight_decay": float(self.config.weight_decay),
                    "schedule": schedule,
                }
                for group in self.parameter_plan.groups
            ]

        plan = {
            "schema_version": 1,
            "candidate": self.candidate,
            "optimizer_class": "torch.optim.adamw.AdamW",
            "static_optimizer_topology": True,
            "initial_phase": "phase_a",
            "trainable_parameter_names": list(
                self.parameter_plan.trainable_parameter_names
            ),
            "campaign_step_range": [0, int(self.config.total_steps) - 1],
            "completion_boundary": int(self.config.total_steps),
            "counter_semantics": {
                "campaign_step": "synchronized successful training batches",
                "optimizer_step": "actual AdamW updates",
            },
            "config": self.config.as_dict(),
            "parameter_plan": self.parameter_plan.as_dict(),
            "parameter_plan_sha256": self.parameter_plan.fingerprint,
            "phases": {
                "phase_a": {
                    "campaign_steps": [0, int(self.config.phase_a_steps) - 1],
                    "optimizer_update": (
                        "adapter_contact" if adapter_phase_a_updates else "none"
                    ),
                    "adapter_schedule": (
                        "linear_warmup_then_constant"
                        if adapter_phase_a_updates
                        else "disabled"
                    ),
                    "base_groups": "disabled_with_grad_none",
                    "groups": planned_groups(
                        phase_a_rates,
                        schedule="linear_warmup_then_constant_for_adapter_only",
                    ),
                },
                "phase_b": {
                    "campaign_steps": [
                        int(self.config.phase_a_steps),
                        int(self.config.total_steps) - 1,
                    ],
                    "schedule": "cosine_to_zero_at_completion_boundary",
                    "adapter_contact_lr": (
                        float(self.config.phase_b_adapter_lr)
                        if adapter_phase_a_updates
                        else 0.0
                    ),
                    "base_lr": float(self.config.phase_b_base_lr),
                    "groups": planned_groups(
                        phase_b_rates,
                        schedule=(
                            "cosine_to_zero_at_campaign_step_"
                            f"{int(self.config.total_steps)}"
                        ),
                    ),
                },
            },
        }
        if self.config.schedule_mode == "warmup_hold_cosine":
            plan["initial_phase"] = "phase_b"
            plan["phases"] = {"phase_b": {
                "campaign_steps": [0, int(self.config.total_steps) - 1],
                "schedule": "warmup_hold_cosine",
                "warmup_steps": self.config.warmup_steps,
                "hold_until_step": self.config.hold_steps,
                "min_lr_ratio": self.config.min_lr_ratio,
                "adapter_contact_lr": self.config.phase_b_adapter_lr if adapter_phase_a_updates else 0.0,
                "base_lr": self.config.phase_b_base_lr,
                "groups": planned_groups(phase_a_rates, schedule="warmup_hold_cosine"),
            }}
        return plan

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "candidate": self.candidate,
            "config": self.config.as_dict(),
            "parameter_plan": self.parameter_plan.as_dict(),
            "parameter_plan_sha256": self.parameter_plan.fingerprint,
            "campaign_step": int(self.campaign_step),
            "optimizer_step": int(self.optimizer_step),
            "optimizer_phase": self.optimizer_phase,
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Strictly restore counters only when every causal field matches."""

        if not isinstance(state_dict, Mapping):
            raise TypeError("score-SE3 controller state must be a mapping")
        required = {
            "schema_version",
            "candidate",
            "config",
            "parameter_plan",
            "parameter_plan_sha256",
            "campaign_step",
            "optimizer_step",
            "optimizer_phase",
        }
        if set(state_dict) != required:
            raise RuntimeError(
                "score-SE3 controller state fields differ: "
                f"missing={sorted(required - set(state_dict))}, "
                f"extra={sorted(set(state_dict) - required)}"
            )
        if int(state_dict["schema_version"]) != self.STATE_SCHEMA_VERSION:
            raise RuntimeError("score-SE3 controller state schema mismatch")
        if str(state_dict["candidate"]) != self.candidate:
            raise RuntimeError("score-SE3 controller candidate mismatch")
        if state_dict["config"] != self.config.as_dict():
            raise RuntimeError("score-SE3 controller schedule/config mismatch")
        if state_dict["parameter_plan"] != self.parameter_plan.as_dict():
            raise RuntimeError("score-SE3 controller parameter plan mismatch")
        if str(state_dict["parameter_plan_sha256"]) != self.parameter_plan.fingerprint:
            raise RuntimeError("score-SE3 controller parameter-plan fingerprint mismatch")

        campaign_step = self._validate_counter(
            state_dict["campaign_step"],
            "campaign_step",
            maximum=self.config.total_steps,
        )
        optimizer_step = self._validate_counter(
            state_dict["optimizer_step"],
            "optimizer_step",
            maximum=campaign_step,
        )
        derived_phase = self.phase_at(campaign_step)
        if str(state_dict["optimizer_phase"]) != derived_phase:
            raise RuntimeError("score-SE3 controller phase/campaign-step mismatch")

        old_campaign_step, old_optimizer_step = self.campaign_step, self.optimizer_step
        self.campaign_step, self.optimizer_step = campaign_step, optimizer_step
        try:
            self._validate_counter_relationship()
            if optimizer is not None:
                self.apply_learning_rates(optimizer)
        except Exception:
            self.campaign_step, self.optimizer_step = old_campaign_step, old_optimizer_step
            raise


@dataclass(frozen=True)
class ScoreSE3FineTuneSetup:
    optimizer: torch.optim.AdamW
    controller: ScoreSE3FineTuneController
    parameter_plan: ScoreSE3ParameterPlan


def prepare_score_se3_finetuning(
    model: nn.Module,
    candidate: str,
    *,
    config: ScoreSE3FineTuneConfig | None = None,
    campaign_step: int = 0,
    optimizer_step: int = 0,
) -> ScoreSE3FineTuneSetup:
    """Freeze, group, build AdamW, and install the initial staged LRs."""

    config = config or ScoreSE3FineTuneConfig()
    # The trainer still routes through the score-SE3 lifecycle because every
    # graph condition inherits metric_pose.  Use the graph identity for optimizer
    # topology and scheduling without changing the checkpoint's metric_pose identity.
    effective_candidate = str(
        getattr(model, "deformation_graph_candidate", candidate)
    )
    plan = build_score_se3_parameter_plan(model, effective_candidate)
    optimizer = build_score_se3_optimizer(model, plan, config)
    controller = ScoreSE3FineTuneController(
        effective_candidate,
        plan,
        config,
        campaign_step=campaign_step,
        optimizer_step=optimizer_step,
    )
    controller.apply_learning_rates(optimizer)
    return ScoreSE3FineTuneSetup(optimizer, controller, plan)


def prepare_score_se3_finetuning_for_resume(
    model: nn.Module,
    candidate: str,
    checkpoint: Mapping[str, Any],
    *,
    config: ScoreSE3FineTuneConfig | None = None,
) -> ScoreSE3FineTuneSetup:
    """Recreate the exact static optimizer/controller at a checkpoint boundary.

    The optimizer state itself is intentionally not loaded here; the shared
    checkpoint loader does that after it validates named-group topology.  This
    helper installs the correct phase/counters first so even a Phase-B resume
    cannot be mistaken for a fresh Phase-A run.
    """

    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    required = {
        "optimizer_step",
        "score_se3_campaign_step",
        "optimizer_phase",
        "lr_scheduler",
    }
    missing = required - set(checkpoint)
    if missing:
        raise RuntimeError(
            f"score-SE3 resume checkpoint is missing {sorted(missing)}"
        )
    campaign_step = checkpoint["score_se3_campaign_step"]
    optimizer_step = checkpoint["optimizer_step"]
    setup = prepare_score_se3_finetuning(
        model,
        candidate,
        config=config,
        campaign_step=campaign_step,
        optimizer_step=optimizer_step,
    )
    expected_phase = setup.controller.optimizer_phase
    if str(checkpoint["optimizer_phase"]) != expected_phase:
        raise RuntimeError(
            "score-SE3 resume checkpoint phase/campaign-step mismatch: "
            f"checkpoint={checkpoint['optimizer_phase']!r}, derived={expected_phase!r}"
        )
    saved_plan = checkpoint.get("score_se3_optimizer_plan")
    if saved_plan is not None and saved_plan != setup.controller.optimizer_plan():
        raise RuntimeError("score-SE3 resume optimizer plan differs from current code")
    scheduler_state = checkpoint["lr_scheduler"]
    if not isinstance(scheduler_state, Mapping):
        raise RuntimeError("score-SE3 resume checkpoint has no controller state")
    setup.controller.load_state_dict(
        scheduler_state, optimizer=setup.optimizer
    )
    if (
        setup.controller.campaign_step != int(campaign_step)
        or setup.controller.optimizer_step != int(optimizer_step)
    ):
        raise RuntimeError("score-SE3 checkpoint counters disagree with controller state")
    return setup


def optimizer_step_performed_after_grad_scaler(
    *,
    scale_before_update: float,
    scale_after_update: float,
) -> bool:
    """Infer whether GradScaler skipped AdamW because of non-finite gradients.

    A scale decrease is the public signal that ``GradScaler.step`` was skipped.
    A stable or growing scale follows a real optimizer step.  This helper is
    only for an enabled scaler; an ordinary non-AMP ``optimizer.step`` is
    always recorded as performed by the caller.
    """

    before = _finite_positive(scale_before_update, "scale_before_update")
    after = _finite_positive(scale_after_update, "scale_after_update")
    return after >= before


__all__ = [
    "ADAPTER_GROUP",
    "BASE_POSE_GROUP",
    "OBJECT_TIME_GROUP",
    "ScoreSE3FineTuneConfig",
    "ScoreSE3FineTuneController",
    "ScoreSE3FineTuneSetup",
    "ScoreSE3ParameterGroupPlan",
    "ScoreSE3ParameterPlan",
    "build_score_se3_optimizer",
    "build_score_se3_parameter_plan",
    "optimizer_step_performed_after_grad_scaler",
    "prepare_score_se3_finetuning",
    "prepare_score_se3_finetuning_for_resume",
]
