from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .address import ConjunctiveAddress, SparseNeuralPlaceCode
from .context import LatentContextHead, PFCStateToken
from .data import PATH_FAMILY_SPLIT_SIZES, PathFamilySplit
from .memory import FactorizedCovarianceHPC
from .memorymaze3d import PeriodicEgocentricSE2GridEC
from .pfc import WindowTransformerPFC


@dataclass(frozen=True)
class VisualMultiPlaceConfig:
    dim_action: int = 6
    dim_feature: int = 384
    dim_hidden: int = 96
    n_heads: int = 4
    window_size: int = 32
    max_sequence_length: int = 128
    dim_state: int = 16
    state_heads: int = 4
    state_window_size: int = 64
    dim_context: int = 8
    dim_structure: int = 32
    dim_place: int = 64
    place_temperature: float = 0.25
    covariance_ridge: float = 0.03
    pfc_residual_gain: float = 0.0
    n_places: int = 4
    cue_length: int = 10
    route_steps: int = 24
    cue_padding: int = 4

    def __post_init__(self) -> None:
        if self.dim_action != 6:
            raise ValueError("MemoryMaze3D uses the official six-action set")
        if self.route_steps <= 0 or self.route_steps % self.n_places:
            raise ValueError("route_steps must be divisible by n_places")
        if self.cue_length < 6:
            raise ValueError("cue_length must leave room for balanced turns")
        if self.max_sequence_length < 3 * (
            self.cue_length + self.cue_padding + self.route_steps
        ):
            raise ValueError("max_sequence_length is too short for A-B-A")


@dataclass(frozen=True)
class VisualContextRouteFamily:
    family_id: int
    canonical_hash: str
    cue_actions: tuple[int, ...]


@dataclass(frozen=True)
class VisualObjectAssignmentFamily:
    family_id: int
    canonical_hash: str
    first_labels: tuple[int, ...]
    second_labels: tuple[int, ...]


@dataclass(frozen=True)
class VisualMultiPlaceBatch:
    actions: torch.Tensor
    observed_features: torch.Tensor
    targets: torch.Tensor
    competing_targets: torch.Tensor
    target_labels: torch.Tensor
    competing_labels: torch.Tensor
    write_mask: torch.Tensor
    query_mask: torch.Tensor
    conflict_mask: torch.Tensor
    clean_mask: torch.Tensor
    first_exposure_mask: torch.Tensor
    event_mask: torch.Tensor
    context_ids: torch.Tensor
    place_ids: torch.Tensor
    segment_ids: torch.Tensor
    route_family_ids: torch.Tensor
    assignment_family_ids: torch.Tensor
    environment_ids: torch.Tensor
    split: str

    def model_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actions, self.observed_features


@dataclass(frozen=True)
class VisualMultiPlaceOutput:
    prediction: torch.Tensor
    pfc_prediction: torch.Tensor
    memory_prediction: torch.Tensor
    pfc_hidden: torch.Tensor
    state: torch.Tensor
    context: torch.Tensor
    structural_code: torch.Tensor
    place_code: torch.Tensor
    address: torch.Tensor
    retrieval: torch.Tensor
    write_count: int
    future_ground_truth_reads: int
    future_ground_truth_writes: int


@lru_cache(maxsize=None)
def build_visual_context_route_bank(
    seed: int = 20_260_726,
    cue_length: int = 10,
) -> dict[PathFamilySplit, tuple[VisualContextRouteFamily, ...]]:
    """Hash-disjoint, zero-net-turn route cues for hidden visual contexts."""

    if cue_length < 6:
        raise ValueError("cue_length must be at least six")
    cues: list[tuple[int, ...]] = []
    positions = tuple(range(cue_length))
    for left_positions in itertools.combinations(positions, 3):
        remaining = tuple(
            position for position in positions if position not in left_positions
        )
        for right_positions in itertools.combinations(remaining, 3):
            cue = [0] * cue_length
            for position in left_positions:
                cue[position] = 2
            for position in right_positions:
                cue[position] = 3
            cues.append(tuple(cue))
    rng = random.Random(seed)
    rng.shuffle(cues)
    total = sum(PATH_FAMILY_SPLIT_SIZES.values())
    if len(cues) < total:
        raise RuntimeError("not enough balanced visual context route families")

    families = []
    for family_id, cue in enumerate(cues[:total]):
        digest = hashlib.sha256(bytes(cue)).hexdigest()[:16]
        families.append(
            VisualContextRouteFamily(
                family_id=family_id,
                canonical_hash=digest,
                cue_actions=cue,
            )
        )

    bank: dict[PathFamilySplit, tuple[VisualContextRouteFamily, ...]] = {}
    start = 0
    for split, count in PATH_FAMILY_SPLIT_SIZES.items():
        bank[split] = tuple(families[start : start + count])
        start += count
    return bank


@lru_cache(maxsize=None)
def build_visual_object_assignment_bank(
    n_places: int = 4,
    n_labels: int = 3,
    seed: int = 20_260_727,
) -> dict[PathFamilySplit, tuple[VisualObjectAssignmentFamily, ...]]:
    """Disjoint context-by-place object mappings for each data split."""

    if n_places <= 0 or n_labels < 2:
        raise ValueError("object assignment dimensions are invalid")
    total = sum(PATH_FAMILY_SPLIT_SIZES.values())
    rng = random.Random(seed)
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    assignments: list[VisualObjectAssignmentFamily] = []
    while len(assignments) < total:
        first = tuple(rng.randrange(n_labels) for _ in range(n_places))
        offsets = tuple(rng.randrange(1, n_labels) for _ in range(n_places))
        second = tuple(
            (label + offset) % n_labels
            for label, offset in zip(first, offsets)
        )
        canonical = min((first, second), (second, first))
        if canonical in seen:
            continue
        seen.add(canonical)
        digest = hashlib.sha256(
            bytes((*canonical[0], 255, *canonical[1]))
        ).hexdigest()[:16]
        assignments.append(
            VisualObjectAssignmentFamily(
                family_id=len(assignments),
                canonical_hash=digest,
                first_labels=first,
                second_labels=second,
            )
        )

    bank: dict[PathFamilySplit, tuple[VisualObjectAssignmentFamily, ...]] = {}
    start = 0
    for split, count in PATH_FAMILY_SPLIT_SIZES.items():
        bank[split] = tuple(assignments[start : start + count])
        start += count
    return bank


def _one_hot_action(index: int, *, dtype: torch.dtype) -> torch.Tensor:
    return F.one_hot(torch.tensor(index), num_classes=6).to(dtype=dtype)


def _sample_index(
    count: int,
    generator: torch.Generator,
) -> int:
    return int(torch.randint(0, count, (), generator=generator).item())


def _choose_feature(
    feature_bank: torch.Tensor,
    feature_labels: torch.Tensor,
    feature_environment_ids: torch.Tensor,
    *,
    environment_id: int,
    label: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    candidates = torch.nonzero(
        (feature_environment_ids == environment_id)
        & (feature_labels == label),
        as_tuple=False,
    ).flatten()
    if candidates.numel() == 0:
        raise ValueError(
            f"environment {environment_id} has no feature for label {label}"
        )
    selected = int(
        candidates[_sample_index(candidates.numel(), generator)].item()
    )
    return feature_bank[selected], selected


def make_visual_multiplace_aba_batch(
    feature_bank: torch.Tensor,
    feature_labels: torch.Tensor,
    feature_environment_ids: torch.Tensor,
    neutral_features: torch.Tensor,
    neutral_environment_ids: torch.Tensor,
    *,
    batch_size: int,
    split: PathFamilySplit,
    device: torch.device,
    generator: torch.Generator,
    config: VisualMultiPlaceConfig = VisualMultiPlaceConfig(),
    conflict_probability: float = 0.75,
) -> VisualMultiPlaceBatch:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= conflict_probability <= 1.0:
        raise ValueError("conflict_probability must lie in [0,1]")
    if feature_bank.ndim != 2 or feature_bank.shape[-1] != config.dim_feature:
        raise ValueError("feature bank has the wrong shape")
    if neutral_features.ndim != 2 or neutral_features.shape[-1] != config.dim_feature:
        raise ValueError("neutral feature bank has the wrong shape")

    feature_bank = F.normalize(feature_bank.detach().cpu(), dim=-1)
    feature_labels = feature_labels.detach().cpu().long()
    feature_environment_ids = feature_environment_ids.detach().cpu().long()
    neutral_features = F.normalize(neutral_features.detach().cpu(), dim=-1)
    neutral_environment_ids = neutral_environment_ids.detach().cpu().long()
    route_bank = build_visual_context_route_bank(
        cue_length=config.cue_length
    )[split]
    assignment_bank = build_visual_object_assignment_bank(
        config.n_places,
        int(feature_labels.max().item()) + 1,
    )[split]
    environments = sorted(
        set(feature_environment_ids.tolist())
        & set(neutral_environment_ids.tolist())
    )
    if not environments:
        raise ValueError("feature and neutral banks share no environments")

    phase_length = config.cue_length + config.cue_padding + config.route_steps
    sequence_length = 3 * phase_length
    event_stride = config.route_steps // config.n_places

    rows: dict[str, list[torch.Tensor]] = {
        "actions": [],
        "observed": [],
        "targets": [],
        "competing": [],
        "target_labels": [],
        "competing_labels": [],
        "write": [],
        "query": [],
        "conflict": [],
        "clean": [],
        "first": [],
        "event": [],
        "context": [],
        "place": [],
        "segment": [],
        "route_family": [],
        "assignment_family": [],
        "environment": [],
    }

    for _ in range(batch_size):
        environment_id = environments[_sample_index(len(environments), generator)]
        neutral_candidates = torch.nonzero(
            neutral_environment_ids == environment_id,
            as_tuple=False,
        ).flatten()
        neutral = neutral_features[
            neutral_candidates[_sample_index(neutral_candidates.numel(), generator)]
        ]
        selected_routes = torch.randperm(
            len(route_bank), generator=generator
        )[:2].tolist()
        route_by_context = (
            route_bank[selected_routes[0]],
            route_bank[selected_routes[1]],
        )
        assignment = assignment_bank[_sample_index(len(assignment_bank), generator)]
        first_context = _sample_index(2, generator)
        context_sequence = (first_context, 1 - first_context, first_context)
        route_action = 4 + _sample_index(2, generator)

        conflict_flags = (
            torch.rand(config.n_places, generator=generator)
            < conflict_probability
        )
        if config.n_places > 1:
            if not conflict_flags.any():
                conflict_flags[_sample_index(config.n_places, generator)] = True
            if conflict_flags.all():
                conflict_flags[_sample_index(config.n_places, generator)] = False

        context_labels = (
            assignment.first_labels,
            assignment.second_labels,
        )
        content_features: dict[tuple[int, int], torch.Tensor] = {}
        episode_labels = [list(labels) for labels in context_labels]
        for place_id in range(config.n_places):
            first_feature, _ = _choose_feature(
                feature_bank,
                feature_labels,
                feature_environment_ids,
                environment_id=environment_id,
                label=context_labels[0][place_id],
                generator=generator,
            )
            if bool(conflict_flags[place_id].item()):
                second_feature, _ = _choose_feature(
                    feature_bank,
                    feature_labels,
                    feature_environment_ids,
                    environment_id=environment_id,
                    label=context_labels[1][place_id],
                    generator=generator,
                )
            else:
                second_feature = first_feature
                episode_labels[1][place_id] = episode_labels[0][place_id]
            content_features[(0, place_id)] = first_feature
            content_features[(1, place_id)] = second_feature

        actions = _one_hot_action(
            0, dtype=feature_bank.dtype
        ).unsqueeze(0).expand(sequence_length, -1).clone()
        observed = neutral.unsqueeze(0).expand(sequence_length, -1).clone()
        targets = observed.clone()
        competing = observed.clone()
        target_labels = torch.full((sequence_length,), -1, dtype=torch.long)
        competing_labels = torch.full(
            (sequence_length,), -1, dtype=torch.long
        )
        write = torch.zeros(sequence_length, 1)
        query = torch.zeros(sequence_length, dtype=torch.bool)
        conflict = torch.zeros(sequence_length, dtype=torch.bool)
        clean = torch.zeros(sequence_length, dtype=torch.bool)
        first = torch.zeros(sequence_length, dtype=torch.bool)
        event = torch.zeros(sequence_length, dtype=torch.bool)
        context = torch.full((sequence_length,), -1, dtype=torch.long)
        place = torch.full((sequence_length,), -1, dtype=torch.long)
        segment = torch.full((sequence_length,), -1, dtype=torch.long)
        route_family = torch.full((sequence_length,), -1, dtype=torch.long)
        assignment_family = torch.full(
            (sequence_length,), assignment.family_id, dtype=torch.long
        )
        environment = torch.full(
            (sequence_length,), environment_id, dtype=torch.long
        )

        seen: set[tuple[int, int]] = set()
        seen_contexts: set[int] = set()
        for phase_id, context_id in enumerate(context_sequence):
            phase_start = phase_id * phase_length
            pre_padding = _sample_index(config.cue_padding + 1, generator)
            cue = route_by_context[context_id]
            cue_start = phase_start + pre_padding
            route_start = (
                phase_start + config.cue_length + config.cue_padding
            )
            context[phase_start : phase_start + phase_length] = context_id
            segment[phase_start : phase_start + phase_length] = phase_id
            route_family[phase_start : phase_start + phase_length] = cue.family_id

            for local_index, action_id in enumerate(cue.cue_actions):
                actions[cue_start + local_index] = _one_hot_action(
                    action_id, dtype=actions.dtype
                )
            for local_index in range(config.route_steps):
                actions[route_start + local_index] = _one_hot_action(
                    route_action, dtype=actions.dtype
                )

            for place_id in range(config.n_places):
                time_index = route_start + (place_id + 1) * event_stride - 1
                key = (context_id, place_id)
                other_key = (1 - context_id, place_id)
                first_exposure = key not in seen
                both_contexts_seen = len(seen_contexts | {context_id}) == 2
                is_query = (not first_exposure) and both_contexts_seen
                target = content_features[key]
                other_target = content_features[other_key]
                event[time_index] = True
                place[time_index] = place_id
                targets[time_index] = target
                competing[time_index] = other_target
                target_labels[time_index] = episode_labels[context_id][place_id]
                competing_labels[time_index] = episode_labels[
                    1 - context_id
                ][place_id]
                first[time_index] = first_exposure
                query[time_index] = is_query
                conflict[time_index] = is_query and bool(
                    conflict_flags[place_id].item()
                )
                clean[time_index] = is_query and not bool(
                    conflict_flags[place_id].item()
                )
                if first_exposure:
                    write[time_index] = 1.0
                    observed[time_index] = target
                    seen.add(key)
            seen_contexts.add(context_id)

        rows["actions"].append(actions)
        rows["observed"].append(observed)
        rows["targets"].append(targets)
        rows["competing"].append(competing)
        rows["target_labels"].append(target_labels)
        rows["competing_labels"].append(competing_labels)
        rows["write"].append(write)
        rows["query"].append(query)
        rows["conflict"].append(conflict)
        rows["clean"].append(clean)
        rows["first"].append(first)
        rows["event"].append(event)
        rows["context"].append(context)
        rows["place"].append(place)
        rows["segment"].append(segment)
        rows["route_family"].append(route_family)
        rows["assignment_family"].append(assignment_family)
        rows["environment"].append(environment)

    return VisualMultiPlaceBatch(
        actions=torch.stack(rows["actions"]).to(device),
        observed_features=torch.stack(rows["observed"]).to(device),
        targets=torch.stack(rows["targets"]).to(device),
        competing_targets=torch.stack(rows["competing"]).to(device),
        target_labels=torch.stack(rows["target_labels"]).to(device),
        competing_labels=torch.stack(rows["competing_labels"]).to(device),
        write_mask=torch.stack(rows["write"]).to(device),
        query_mask=torch.stack(rows["query"]).to(device),
        conflict_mask=torch.stack(rows["conflict"]).to(device),
        clean_mask=torch.stack(rows["clean"]).to(device),
        first_exposure_mask=torch.stack(rows["first"]).to(device),
        event_mask=torch.stack(rows["event"]).to(device),
        context_ids=torch.stack(rows["context"]).to(device),
        place_ids=torch.stack(rows["place"]).to(device),
        segment_ids=torch.stack(rows["segment"]).to(device),
        route_family_ids=torch.stack(rows["route_family"]).to(device),
        assignment_family_ids=torch.stack(rows["assignment_family"]).to(device),
        environment_ids=torch.stack(rows["environment"]).to(device),
        split=split,
    )


class VisualMultiPlaceReMAPFormer(nn.Module):
    """Visual A-B-A with an action-only neural EC and episode-local HPC."""

    def __init__(
        self,
        config: VisualMultiPlaceConfig = VisualMultiPlaceConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.pfc = WindowTransformerPFC(
            dim_action=config.dim_action,
            dim_sensory=config.dim_feature,
            dim_hidden=config.dim_hidden,
            n_heads=config.n_heads,
            window_size=config.window_size,
            max_sequence_length=config.max_sequence_length,
        )
        self.state_token = PFCStateToken(
            dim_input=config.dim_hidden,
            dim_action=config.dim_action,
            dim_state=config.dim_state,
            n_heads=config.state_heads,
            window_size=config.state_window_size,
            architecture="pfc_history_retention_v6",
        )
        self.context_head = LatentContextHead(
            dim_state=config.dim_state,
            dim_context=config.dim_context,
        )
        self.ec = PeriodicEgocentricSE2GridEC(config.dim_structure)
        self.place = SparseNeuralPlaceCode(
            config.dim_structure,
            config.dim_place,
            config.place_temperature,
        )
        for parameter in self.ec.parameters():
            parameter.requires_grad_(False)
        for parameter in self.place.parameters():
            parameter.requires_grad_(False)
        self.address = ConjunctiveAddress(config.dim_context)
        self.hpc = FactorizedCovarianceHPC(
            config.dim_feature,
            config.dim_feature,
            dim_context=config.dim_context,
            covariance_ridge=config.covariance_ridge,
        )
        self.hpc.value_encoder.weight.requires_grad_(False)
        self.hpc.value_decoder.weight.requires_grad_(False)

    def forward(
        self,
        actions: torch.Tensor,
        observed_features: torch.Tensor,
        *,
        write_mask: torch.Tensor,
        context_override: torch.Tensor | None = None,
        disable_hpc_read: bool = False,
    ) -> VisualMultiPlaceOutput:
        pfc_prediction, pfc_hidden = self.pfc(
            actions,
            observed_features,
            return_hidden=True,
        )
        state = self.state_token(pfc_hidden, actions=actions)
        context = self.context_head(state)
        if context_override is not None:
            if context_override.shape != context.shape:
                raise ValueError("context override has the wrong shape")
            context = F.normalize(context_override, dim=-1)
        structural = self.ec(actions)
        place = self.place(structural)
        address = self.address(place, context)
        hpc_write_mask = (
            write_mask.squeeze(-1)
            if write_mask.ndim == 3 and write_mask.shape[-1] == 1
            else write_mask
        )
        memory = self.hpc(
            address,
            observed_features,
            write_gate=hpc_write_mask,
        )
        retrieval = memory.retrieval
        if disable_hpc_read:
            retrieval = torch.zeros_like(retrieval)
        memory_prediction = self.hpc.value_decoder(retrieval)
        prediction = (
            memory_prediction
            + self.config.pfc_residual_gain * pfc_prediction
        )
        return VisualMultiPlaceOutput(
            prediction=prediction,
            pfc_prediction=pfc_prediction,
            memory_prediction=memory_prediction,
            pfc_hidden=pfc_hidden,
            state=state,
            context=context,
            structural_code=structural,
            place_code=place,
            address=address,
            retrieval=retrieval,
            write_count=int(write_mask.sum().item()),
            future_ground_truth_reads=0,
            future_ground_truth_writes=0,
        )


def visual_multiplace_content_loss(
    prediction: torch.Tensor,
    batch: VisualMultiPlaceBatch,
    *,
    temperature: float = 0.05,
    clean_weight: float = 1.0,
) -> torch.Tensor:
    losses = []
    if batch.conflict_mask.any():
        predicted = F.normalize(prediction[batch.conflict_mask], dim=-1)
        correct = F.normalize(batch.targets[batch.conflict_mask], dim=-1)
        competing = F.normalize(
            batch.competing_targets[batch.conflict_mask], dim=-1
        )
        logits = torch.stack(
            (
                (predicted * correct).sum(dim=-1),
                (predicted * competing).sum(dim=-1),
            ),
            dim=-1,
        )
        losses.append(
            F.cross_entropy(
                logits / temperature,
                torch.zeros(
                    logits.shape[0],
                    dtype=torch.long,
                    device=logits.device,
                ),
            )
        )
    if batch.clean_mask.any():
        cosine = F.cosine_similarity(
            prediction[batch.clean_mask],
            batch.targets[batch.clean_mask],
            dim=-1,
        )
        losses.append(clean_weight * (1.0 - cosine.mean()))
    if not losses:
        raise ValueError("visual multi-place loss has no query samples")
    return torch.stack(losses).sum()


def context_id_oracle(
    batch: VisualMultiPlaceBatch,
    dim_context: int,
) -> torch.Tensor:
    if dim_context < 2:
        raise ValueError("orthogonal context oracle requires two dimensions")
    return F.one_hot(
        batch.context_ids,
        num_classes=dim_context,
    ).to(dtype=batch.targets.dtype)


def build_multiplace_context_overrides(
    context: torch.Tensor,
    batch: VisualMultiPlaceBatch,
) -> dict[str, torch.Tensor]:
    correct = context.clone()
    wrong = context.clone()
    for batch_index in range(context.shape[0]):
        first_indices: dict[int, int] = {}
        exposure_indices = torch.nonzero(
            batch.first_exposure_mask[batch_index],
            as_tuple=False,
        ).flatten()
        for index in exposure_indices:
            time_index = int(index.item())
            context_id = int(
                batch.context_ids[batch_index, time_index].item()
            )
            first_indices.setdefault(context_id, time_index)
        query_indices = torch.nonzero(
            batch.query_mask[batch_index],
            as_tuple=False,
        ).flatten()
        for index in query_indices:
            time_index = int(index.item())
            context_id = int(
                batch.context_ids[batch_index, time_index].item()
            )
            correct[batch_index, time_index] = context[
                batch_index, first_indices[context_id]
            ]
            wrong[batch_index, time_index] = context[
                batch_index, first_indices[1 - context_id]
            ]
    fixed = torch.ones_like(context)
    return {
        "correct_history": F.normalize(correct, dim=-1),
        "wrong_history": F.normalize(wrong, dim=-1),
        "fixed_context": F.normalize(fixed, dim=-1),
    }


def event_place_geometry(
    place_code: torch.Tensor,
    batch: VisualMultiPlaceBatch,
) -> dict[str, float]:
    same_place = []
    cross_place = []
    for batch_index in range(place_code.shape[0]):
        event_indices = torch.nonzero(
            batch.event_mask[batch_index],
            as_tuple=False,
        ).flatten()
        for left_pos, left in enumerate(event_indices.tolist()):
            for right in event_indices[left_pos + 1 :].tolist():
                left_place = int(batch.place_ids[batch_index, left].item())
                right_place = int(batch.place_ids[batch_index, right].item())
                left_segment = int(
                    batch.segment_ids[batch_index, left].item()
                )
                right_segment = int(
                    batch.segment_ids[batch_index, right].item()
                )
                similarity = F.cosine_similarity(
                    place_code[batch_index, left],
                    place_code[batch_index, right],
                    dim=0,
                )
                if left_place == right_place and left_segment != right_segment:
                    same_place.append(similarity)
                elif (
                    left_place != right_place
                    and left_segment == right_segment
                ):
                    cross_place.append(similarity)
    if not same_place or not cross_place:
        raise ValueError("event place geometry has empty comparison groups")
    return {
        "same_place_cross_context_cosine": float(
            torch.stack(same_place).mean().item()
        ),
        "same_segment_cross_place_cosine": float(
            torch.stack(cross_place).mean().item()
        ),
    }
