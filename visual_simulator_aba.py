from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import PathFamilySplit
from .visual_multiplace_aba import (
    VisualMultiPlaceConfig,
    VisualMultiPlaceReMAPFormer,
)


@dataclass(frozen=True)
class SimulatorABABatch:
    actions: torch.Tensor
    observed_features: torch.Tensor
    targets: torch.Tensor
    competing_targets: torch.Tensor
    target_labels: torch.Tensor
    competing_labels: torch.Tensor
    write_mask: torch.Tensor
    query_mask: torch.Tensor
    delayed_query_mask: torch.Tensor
    conflict_mask: torch.Tensor
    clean_mask: torch.Tensor
    anchor_mask: torch.Tensor
    event_mask: torch.Tensor
    context_ids: torch.Tensor
    place_ids: torch.Tensor
    segment_ids: torch.Tensor
    route_family_ids: torch.Tensor
    target_write_indices: torch.Tensor
    competing_write_indices: torch.Tensor
    environment_ids: torch.Tensor
    episode_indices: torch.Tensor
    split: str

    def model_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actions, self.observed_features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SimulatorABADataset:
    """Tensor view of simulator episodes; images and metadata stay out of inputs."""

    METADATA_FIELDS = (
        "target_labels",
        "competing_labels",
        "write_mask",
        "query_mask",
        "delayed_query_mask",
        "conflict_mask",
        "clean_mask",
        "anchor_mask",
        "event_mask",
        "context_ids",
        "place_ids",
        "segment_ids",
        "route_family_ids",
        "target_write_indices",
        "competing_write_indices",
        "visible_target_labels",
        "target_geom_radius",
    )

    def __init__(
        self,
        data_path: str | Path,
        cache_path: str | Path,
        *,
        split: str,
    ) -> None:
        self.data_path = Path(data_path)
        self.cache_path = Path(cache_path)
        self.split = split
        cache = torch.load(
            self.cache_path,
            map_location="cpu",
            weights_only=False,
        )
        if cache["source"]["sha256"] != _sha256(self.data_path):
            raise ValueError("DINO cache does not match simulator data")
        self.features = F.normalize(
            cache["frame_features"].float(),
            dim=-1,
        )
        arrays: dict[str, torch.Tensor] = {}
        with np.load(self.data_path, allow_pickle=False) as source:
            actions = torch.from_numpy(
                source["actions"].astype(np.int64, copy=True)
            )
            for field in self.METADATA_FIELDS:
                arrays[field] = torch.from_numpy(
                    np.array(source[field], copy=True)
                )
        if actions.shape != self.features.shape[:2]:
            raise ValueError("action and DINO sequence shapes do not align")
        if torch.any((actions < 0) | (actions >= 6)):
            raise ValueError("action outside the official six-action set")
        self.actions = F.one_hot(actions, num_classes=6).float()
        for name, value in arrays.items():
            setattr(self, name, value)
        self.episode_count = int(self.features.shape[0])
        self.sequence_length = int(self.features.shape[1])
        self.dim_feature = int(self.features.shape[2])

    def __len__(self) -> int:
        return self.episode_count

    @staticmethod
    def _historical_targets(
        features: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        safe = indices.long().clamp_min(0)
        gathered = torch.gather(
            features,
            1,
            safe.unsqueeze(-1).expand(-1, -1, features.shape[-1]),
        )
        return torch.where(
            (indices >= 0).unsqueeze(-1),
            gathered,
            torch.zeros_like(gathered),
        )

    def batch(
        self,
        episode_indices: torch.Tensor,
        *,
        device: torch.device,
    ) -> SimulatorABABatch:
        episode_indices = episode_indices.long().cpu()
        observed = self.features[episode_indices].to(device)
        target_write_indices = self.target_write_indices[
            episode_indices
        ].to(device)
        competing_write_indices = self.competing_write_indices[
            episode_indices
        ].to(device)
        environment_ids = episode_indices[:, None].expand(
            -1,
            self.sequence_length,
        )
        return SimulatorABABatch(
            actions=self.actions[episode_indices].to(device),
            observed_features=observed,
            targets=self._historical_targets(
                observed,
                target_write_indices,
            ),
            competing_targets=self._historical_targets(
                observed,
                competing_write_indices,
            ),
            target_labels=self.target_labels[episode_indices].long().to(device),
            competing_labels=self.competing_labels[
                episode_indices
            ].long().to(device),
            write_mask=self.write_mask[episode_indices]
            .float()
            .unsqueeze(-1)
            .to(device),
            query_mask=self.query_mask[episode_indices].bool().to(device),
            delayed_query_mask=self.delayed_query_mask[
                episode_indices
            ].bool().to(device),
            conflict_mask=self.conflict_mask[episode_indices].bool().to(device),
            clean_mask=self.clean_mask[episode_indices].bool().to(device),
            anchor_mask=self.anchor_mask[episode_indices].bool().to(device),
            event_mask=self.event_mask[episode_indices].bool().to(device),
            context_ids=self.context_ids[episode_indices].long().to(device),
            place_ids=self.place_ids[episode_indices].long().to(device),
            segment_ids=self.segment_ids[episode_indices].long().to(device),
            route_family_ids=self.route_family_ids[
                episode_indices
            ].long().to(device),
            target_write_indices=target_write_indices,
            competing_write_indices=competing_write_indices,
            environment_ids=environment_ids.to(device),
            episode_indices=episode_indices.to(device),
            split=self.split,
        )

    def all(self, *, device: torch.device) -> SimulatorABABatch:
        return self.batch(torch.arange(len(self)), device=device)


class SimulatorCoupledViewPlaceReMAPFormer(VisualMultiPlaceReMAPFormer):
    """ReMAP-Former calibrated to the simulator's 20-turn full rotation."""

    def __init__(
        self,
        config: VisualMultiPlaceConfig = VisualMultiPlaceConfig(),
    ) -> None:
        super().__init__(config)
        with torch.no_grad():
            self.ec.log_turn_scale.fill_(math.log(math.pi / 10.0))
        self.ec.log_turn_scale.requires_grad_(False)


def build_simulator_context_overrides(
    context: torch.Tensor,
    batch: SimulatorABABatch,
) -> dict[str, torch.Tensor]:
    correct = context.clone()
    wrong = context.clone()
    for batch_index in range(context.shape[0]):
        anchor_indices = torch.nonzero(
            batch.anchor_mask[batch_index],
            as_tuple=False,
        ).flatten()
        if anchor_indices.numel() != 1:
            raise ValueError("each simulator episode needs one return anchor")
        anchor_index = int(anchor_indices.item())
        current_context = int(
            batch.context_ids[batch_index, anchor_index].item()
        )
        other_candidates = torch.nonzero(
            (batch.write_mask[batch_index, :, 0] > 0)
            & (batch.context_ids[batch_index] == 1 - current_context)
            & (batch.place_ids[batch_index] == 0),
            as_tuple=False,
        ).flatten()
        if other_candidates.numel() != 1:
            raise ValueError("other context has no unique place-zero anchor")
        other_index = int(other_candidates.item())
        query_indices = torch.nonzero(
            batch.query_mask[batch_index],
            as_tuple=False,
        ).flatten()
        correct[batch_index, query_indices] = context[
            batch_index,
            anchor_index,
        ]
        wrong[batch_index, query_indices] = context[
            batch_index,
            other_index,
        ]
    fixed = torch.ones_like(context)
    return {
        "correct_history": F.normalize(correct, dim=-1),
        "wrong_history": F.normalize(wrong, dim=-1),
        "fixed_context": F.normalize(fixed, dim=-1),
    }


def simulator_context_oracle(
    batch: SimulatorABABatch,
    dim_context: int,
) -> torch.Tensor:
    if dim_context < 2:
        raise ValueError("context oracle needs at least two dimensions")
    return F.one_hot(
        batch.context_ids,
        num_classes=dim_context,
    ).to(dtype=batch.observed_features.dtype)


def simulator_event_place_geometry(
    place_code: torch.Tensor,
    batch: SimulatorABABatch,
) -> dict[str, float]:
    same_place = []
    cross_place = []
    for batch_index in range(place_code.shape[0]):
        event_indices = torch.nonzero(
            batch.event_mask[batch_index],
            as_tuple=False,
        ).flatten()
        for left_position, left in enumerate(event_indices.tolist()):
            for right in event_indices[left_position + 1 :].tolist():
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
        raise ValueError("simulator place geometry has an empty group")
    return {
        "same_place_cross_context_cosine": float(
            torch.stack(same_place).mean().item()
        ),
        "same_segment_cross_place_cosine": float(
            torch.stack(cross_place).mean().item()
        ),
    }


def simulator_input_audit(
    dataset: SimulatorABADataset,
) -> dict[str, int | bool | float]:
    query_indices = torch.nonzero(dataset.query_mask, as_tuple=False)
    target_sources = dataset.target_write_indices[
        dataset.query_mask
    ].long()
    competing_sources = dataset.competing_write_indices[
        dataset.query_mask
    ].long()
    query_times = query_indices[:, 1].long()
    return {
        "sequence_length": dataset.sequence_length,
        "write_count_per_episode_min": int(
            dataset.write_mask.sum(dim=1).min().item()
        ),
        "write_count_per_episode_max": int(
            dataset.write_mask.sum(dim=1).max().item()
        ),
        "query_count_per_episode_min": int(
            dataset.query_mask.sum(dim=1).min().item()
        ),
        "query_count_per_episode_max": int(
            dataset.query_mask.sum(dim=1).max().item()
        ),
        "target_sources_strictly_historical": bool(
            torch.all(target_sources < query_times).item()
        ),
        "competing_sources_strictly_historical": bool(
            torch.all(competing_sources < query_times).item()
        ),
        "query_visible_target_count": int(
            (dataset.visible_target_labels[dataset.query_mask] >= 0).sum().item()
        ),
        "query_nonhidden_geom_count": int(
            (
                dataset.target_geom_radius[dataset.query_mask] > 0.0011
            ).sum().item()
        ),
        "dino_finite": bool(torch.isfinite(dataset.features).all().item()),
        "model_input_tensor_count": len(
            dataset.all(device=torch.device("cpu")).model_inputs()
        ),
        "metadata_tensor_count": len(dataset.METADATA_FIELDS),
    }


def path_split_for_name(name: str) -> PathFamilySplit:
    aliases = {
        "train": "train",
        "val": "validation",
        "stage_gate": "dev",
    }
    try:
        return aliases[name]
    except KeyError as error:
        raise ValueError(f"unsupported simulator split: {name}") from error
