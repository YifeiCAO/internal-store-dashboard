from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from generate_memorymaze3d_simulator_aba_data import (
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    CAMERA_RESOLUTION,
    CUE_LENGTH,
    TARGET_COUNT,
    TargetPresenter,
    _assignment,
    _layout_hash,
    build_environment,
    build_route_bank,
    sha256_file,
    target_visibility_score,
)


ACTION_FORWARD = 1
N_WAYPOINTS = 4
PHASE_STEPS = 128
EPISODE_STEPS = 3 * PHASE_STEPS
MOVE_DISTANCE_THRESHOLD = 0.24
MOVE_FINAL_ERROR_LIMIT = 0.25
REVISIT_ERROR_LIMIT = 0.30
TURN_ACTION_THRESHOLD = 0.16
FINAL_HEADING_ERROR_LIMIT = 0.16
MAX_MOVE_ACTIONS = 50
MAX_ORIENT_ACTIONS = 20
EC_FORWARD_SCALE = 0.25
EC_TURN_SCALE = math.pi / 10.0


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _angle(vector: np.ndarray) -> float:
    return math.atan2(float(vector[1]), float(vector[0]))


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _waypoint_path(
    layout: np.ndarray,
    agent_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    start = (
        int(math.floor(float(agent_position[1]))),
        int(math.floor(float(agent_position[0]))),
    )
    if not (
        0 <= start[0] < layout.shape[0]
        and 0 <= start[1] < layout.shape[1]
        and bool(layout[start])
    ):
        raise RuntimeError("reset agent is not in an open maze cell")
    queue: deque[tuple[int, int]] = deque((start,))
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    farthest = start
    while queue:
        current = queue.popleft()
        farthest = current
        for row_delta, column_delta in (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ):
            neighbor = (
                current[0] + row_delta,
                current[1] + column_delta,
            )
            if (
                0 <= neighbor[0] < layout.shape[0]
                and 0 <= neighbor[1] < layout.shape[1]
                and bool(layout[neighbor])
                and neighbor not in parent
            ):
                parent[neighbor] = current
                queue.append(neighbor)
    path = []
    cursor: tuple[int, int] | None = farthest
    while cursor is not None:
        path.append(cursor)
        cursor = parent[cursor]
    path.reverse()
    if len(path) < N_WAYPOINTS:
        raise RuntimeError("layout has no four-cell translated route")
    cells = np.asarray(path[:N_WAYPOINTS], dtype=np.int16)
    centers = np.stack(
        (
            cells[:, 1].astype(np.float32) + 0.5,
            cells[:, 0].astype(np.float32) + 0.5,
        ),
        axis=-1,
    )
    route_hash = _sha256_bytes(cells.tobytes())[:16]
    return cells, centers, route_hash


def _integrate_action_pose(actions: np.ndarray) -> np.ndarray:
    position = np.zeros(2, dtype=np.float64)
    heading = 0.0
    poses = []
    for action in actions.tolist():
        if action == ACTION_LEFT:
            heading += EC_TURN_SCALE
        elif action == ACTION_RIGHT:
            heading -= EC_TURN_SCALE
        if action == ACTION_FORWARD:
            position += EC_FORWARD_SCALE * np.asarray(
                (math.cos(heading), math.sin(heading)),
                dtype=np.float64,
            )
        poses.append((position[0], position[1], heading))
    return np.asarray(poses, dtype=np.float32)


def _event_pose_table(
    arrays: dict[str, np.ndarray],
    field: str,
) -> np.ndarray:
    source = arrays[field]
    rows = []
    for segment_id in range(3):
        segment_rows = []
        for place_id in range(N_WAYPOINTS):
            mask = (
                arrays["event_mask"]
                & (arrays["segment_ids"] == segment_id)
                & (arrays["place_ids"] == place_id)
            )
            indices = np.flatnonzero(mask)
            if indices.size != 1:
                raise RuntimeError(
                    "each segment/place must contain exactly one write or query"
                )
            segment_rows.append(source[int(indices[0])])
        rows.append(segment_rows)
    return np.asarray(rows)


def generate_episode(
    *,
    environment_seed: int,
    split: str,
    route_rng_seed: int,
    minimum_visibility: int,
    first_context: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if split not in {"train", "val", "stage_gate", "test"}:
        raise ValueError(
            "generation only supports train/val/stage_gate/test"
        )
    if first_context not in (0, 1):
        raise ValueError("first_context must be zero or one")
    environment = build_environment(environment_seed)
    time_step = environment.reset()
    observation = time_step.observation
    presenter = TargetPresenter(environment)
    presenter.hide_all()
    initial_position = np.asarray(
        observation["agent_pos"],
        dtype=np.float64,
    ).copy()
    initial_heading = _angle(
        np.asarray(observation["agent_dir"], dtype=np.float64)
    )
    waypoint_cells, waypoint_centers, physical_route_hash = _waypoint_path(
        np.asarray(observation["maze_layout"], dtype=np.uint8),
        initial_position,
    )
    route_rng = np.random.default_rng(route_rng_seed)
    route_bank = build_route_bank()[split]
    selected = route_rng.choice(len(route_bank), size=2, replace=False)
    routes = (route_bank[int(selected[0])], route_bank[int(selected[1])])
    cue_label_mapping = route_rng.choice(
        TARGET_COUNT,
        size=2,
        replace=False,
    ).astype(np.int64)
    first_labels, second_labels, assignment_hash = _assignment(route_rng)
    labels_by_context = (first_labels, second_labels)
    context_sequence = (first_context, 1 - first_context, first_context)

    rows: dict[str, list[Any]] = {
        "images": [],
        "actions": [],
        "agent_pos": [],
        "agent_dir": [],
        "visible_target_labels": [],
        "visibility_scores": [],
        "target_geom_radius": [],
        "write_mask": [],
        "query_mask": [],
        "delayed_query_mask": [],
        "conflict_mask": [],
        "clean_mask": [],
        "anchor_mask": [],
        "event_mask": [],
        "context_ids": [],
        "place_ids": [],
        "segment_ids": [],
        "route_family_ids": [],
        "target_labels": [],
        "competing_labels": [],
        "target_write_indices": [],
        "competing_write_indices": [],
    }
    write_indices: dict[tuple[int, int], int] = {}
    minimum_observed_visibility = math.inf
    phase_action_counts = []
    phase_closure_errors = []
    phase_heading_errors = []

    def append_step(
        action: int,
        *,
        visible_label: int = -1,
        write: bool = False,
        query: bool = False,
        delayed_query: bool = False,
        anchor: bool = False,
        context_id: int,
        place_id: int = -1,
        segment_id: int,
        route_family_id: int,
        target_label: int = -1,
        competing_label: int = -1,
        target_write_index: int = -1,
        competing_write_index: int = -1,
    ) -> int:
        nonlocal observation, time_step, minimum_observed_visibility
        time_step = environment.step(action)
        observation = time_step.observation
        image = np.asarray(observation["image"], dtype=np.uint8).copy()
        score = (
            target_visibility_score(image, presenter.colors[visible_label])
            if visible_label >= 0
            else 0
        )
        if visible_label >= 0:
            minimum_observed_visibility = min(minimum_observed_visibility, score)
        rows["images"].append(image)
        rows["actions"].append(action)
        rows["agent_pos"].append(
            np.asarray(observation["agent_pos"], dtype=np.float32).copy()
        )
        rows["agent_dir"].append(
            np.asarray(observation["agent_dir"], dtype=np.float32).copy()
        )
        rows["visible_target_labels"].append(visible_label)
        rows["visibility_scores"].append(score)
        rows["target_geom_radius"].append(presenter.max_visible_radius())
        rows["write_mask"].append(write)
        rows["query_mask"].append(query)
        rows["delayed_query_mask"].append(delayed_query)
        rows["conflict_mask"].append(
            query and target_label != competing_label
        )
        rows["clean_mask"].append(
            query and target_label == competing_label
        )
        rows["anchor_mask"].append(anchor)
        rows["event_mask"].append(write or query)
        rows["context_ids"].append(context_id)
        rows["place_ids"].append(place_id)
        rows["segment_ids"].append(segment_id)
        rows["route_family_ids"].append(route_family_id)
        rows["target_labels"].append(target_label)
        rows["competing_labels"].append(competing_label)
        rows["target_write_indices"].append(target_write_index)
        rows["competing_write_indices"].append(competing_write_index)
        return len(rows["actions"]) - 1

    def append_background(
        action: int,
        *,
        context_id: int,
        segment_id: int,
        route_family_id: int,
    ) -> None:
        presenter.hide_all()
        append_step(
            action,
            context_id=context_id,
            segment_id=segment_id,
            route_family_id=route_family_id,
        )

    def move_to(
        goal: np.ndarray,
        *,
        context_id: int,
        segment_id: int,
        route_family_id: int,
    ) -> None:
        for _ in range(MAX_MOVE_ACTIONS):
            delta = goal - np.asarray(
                observation["agent_pos"],
                dtype=np.float64,
            )
            distance = float(np.linalg.norm(delta))
            if distance <= MOVE_DISTANCE_THRESHOLD:
                break
            angle_error = _wrap_angle(
                _angle(delta)
                - _angle(
                    np.asarray(observation["agent_dir"], dtype=np.float64)
                )
            )
            if angle_error > TURN_ACTION_THRESHOLD:
                action = ACTION_LEFT
            elif angle_error < -TURN_ACTION_THRESHOLD:
                action = ACTION_RIGHT
            else:
                action = ACTION_FORWARD
            append_background(
                action,
                context_id=context_id,
                segment_id=segment_id,
                route_family_id=route_family_id,
            )
        else:
            raise RuntimeError("waypoint controller exceeded movement budget")
        for _ in range(2):
            append_background(
                ACTION_NOOP,
                context_id=context_id,
                segment_id=segment_id,
                route_family_id=route_family_id,
            )
        final_error = float(
            np.linalg.norm(
                goal
                - np.asarray(
                    observation["agent_pos"],
                    dtype=np.float64,
                )
            )
        )
        if final_error > MOVE_FINAL_ERROR_LIMIT:
            raise RuntimeError(
                f"waypoint center error {final_error:.4f} exceeds limit"
            )

    def restore_heading(
        *,
        context_id: int,
        segment_id: int,
        route_family_id: int,
    ) -> None:
        for _ in range(MAX_ORIENT_ACTIONS):
            error = _wrap_angle(
                initial_heading
                - _angle(
                    np.asarray(observation["agent_dir"], dtype=np.float64)
                )
            )
            if abs(error) <= 0.10:
                break
            append_background(
                ACTION_LEFT if error > 0 else ACTION_RIGHT,
                context_id=context_id,
                segment_id=segment_id,
                route_family_id=route_family_id,
            )
        else:
            raise RuntimeError("heading controller exceeded action budget")
        append_background(
            ACTION_NOOP,
            context_id=context_id,
            segment_id=segment_id,
            route_family_id=route_family_id,
        )
        final_error = abs(
            _wrap_angle(
                initial_heading
                - _angle(
                    np.asarray(observation["agent_dir"], dtype=np.float64)
                )
            )
        )
        if final_error > FINAL_HEADING_ERROR_LIMIT:
            raise RuntimeError(
                f"phase heading error {final_error:.4f} exceeds limit"
            )

    for segment_id, context_id in enumerate(context_sequence):
        phase_start = len(rows["actions"])
        route = routes[context_id]
        presenter.hide_all()
        for cue_symbol in route.actions:
            if cue_symbol == ACTION_NOOP:
                presenter.hide_all()
                visible_label = -1
            else:
                visible_label = int(
                    cue_label_mapping[
                        0 if cue_symbol == ACTION_LEFT else 1
                    ]
                )
                presenter.show(visible_label, observation)
            append_step(
                ACTION_NOOP,
                visible_label=visible_label,
                context_id=context_id,
                segment_id=segment_id,
                route_family_id=route.family_id,
            )

        for place_id, goal in enumerate(waypoint_centers):
            move_to(
                goal.astype(np.float64),
                context_id=context_id,
                segment_id=segment_id,
                route_family_id=route.family_id,
            )
            label = int(labels_by_context[context_id][place_id])
            other_label = int(labels_by_context[1 - context_id][place_id])
            if segment_id < 2:
                presenter.show(label, observation)
                write_index = append_step(
                    ACTION_NOOP,
                    visible_label=label,
                    write=True,
                    context_id=context_id,
                    place_id=place_id,
                    segment_id=segment_id,
                    route_family_id=route.family_id,
                    target_label=label,
                    competing_label=other_label,
                )
                write_indices[(context_id, place_id)] = write_index
            elif place_id == 0:
                presenter.show(label, observation)
                append_step(
                    ACTION_NOOP,
                    visible_label=label,
                    anchor=True,
                    context_id=context_id,
                    place_id=place_id,
                    segment_id=segment_id,
                    route_family_id=route.family_id,
                    target_label=label,
                    competing_label=other_label,
                )
                presenter.hide_all()
                append_step(
                    ACTION_NOOP,
                    query=True,
                    context_id=context_id,
                    place_id=place_id,
                    segment_id=segment_id,
                    route_family_id=route.family_id,
                    target_label=label,
                    competing_label=other_label,
                    target_write_index=write_indices[(context_id, place_id)],
                    competing_write_index=write_indices[
                        (1 - context_id, place_id)
                    ],
                )
            else:
                presenter.hide_all()
                append_step(
                    ACTION_NOOP,
                    query=True,
                    delayed_query=True,
                    context_id=context_id,
                    place_id=place_id,
                    segment_id=segment_id,
                    route_family_id=route.family_id,
                    target_label=label,
                    competing_label=other_label,
                    target_write_index=write_indices[(context_id, place_id)],
                    competing_write_index=write_indices[
                        (1 - context_id, place_id)
                    ],
                )
            append_background(
                ACTION_NOOP,
                context_id=context_id,
                segment_id=segment_id,
                route_family_id=route.family_id,
            )

        for goal in waypoint_centers[-2::-1]:
            move_to(
                goal.astype(np.float64),
                context_id=context_id,
                segment_id=segment_id,
                route_family_id=route.family_id,
            )
        restore_heading(
            context_id=context_id,
            segment_id=segment_id,
            route_family_id=route.family_id,
        )
        unpadded_count = len(rows["actions"]) - phase_start
        if unpadded_count > PHASE_STEPS:
            raise RuntimeError(
                f"phase uses {unpadded_count} actions; limit is {PHASE_STEPS}"
            )
        phase_action_counts.append(unpadded_count)
        phase_closure_errors.append(
            float(
                np.linalg.norm(
                    np.asarray(
                        observation["agent_pos"],
                        dtype=np.float64,
                    )
                    - initial_position
                )
            )
        )
        phase_heading_errors.append(
            abs(
                _wrap_angle(
                    initial_heading
                    - _angle(
                        np.asarray(
                            observation["agent_dir"],
                            dtype=np.float64,
                        )
                    )
                )
            )
        )
        while len(rows["actions"]) - phase_start < PHASE_STEPS:
            append_background(
                ACTION_NOOP,
                context_id=context_id,
                segment_id=segment_id,
                route_family_id=route.family_id,
            )

    if hasattr(environment, "close"):
        environment.close()
    if len(rows["actions"]) != EPISODE_STEPS:
        raise RuntimeError(
            f"expected {EPISODE_STEPS} simulator steps, got {len(rows['actions'])}"
        )
    if minimum_observed_visibility < minimum_visibility:
        raise RuntimeError(
            f"visible target score {minimum_observed_visibility} "
            f"is below {minimum_visibility}"
        )

    arrays = {
        "images": np.stack(rows["images"]),
        "actions": np.asarray(rows["actions"], dtype=np.int8),
        "agent_pos": np.stack(rows["agent_pos"]),
        "agent_dir": np.stack(rows["agent_dir"]),
        "visible_target_labels": np.asarray(
            rows["visible_target_labels"], dtype=np.int8
        ),
        "visibility_scores": np.asarray(rows["visibility_scores"], dtype=np.int32),
        "target_geom_radius": np.asarray(
            rows["target_geom_radius"], dtype=np.float32
        ),
        "write_mask": np.asarray(rows["write_mask"], dtype=np.bool_),
        "query_mask": np.asarray(rows["query_mask"], dtype=np.bool_),
        "delayed_query_mask": np.asarray(
            rows["delayed_query_mask"], dtype=np.bool_
        ),
        "conflict_mask": np.asarray(rows["conflict_mask"], dtype=np.bool_),
        "clean_mask": np.asarray(rows["clean_mask"], dtype=np.bool_),
        "anchor_mask": np.asarray(rows["anchor_mask"], dtype=np.bool_),
        "event_mask": np.asarray(rows["event_mask"], dtype=np.bool_),
        "context_ids": np.asarray(rows["context_ids"], dtype=np.int8),
        "place_ids": np.asarray(rows["place_ids"], dtype=np.int8),
        "segment_ids": np.asarray(rows["segment_ids"], dtype=np.int8),
        "route_family_ids": np.asarray(rows["route_family_ids"], dtype=np.int16),
        "target_labels": np.asarray(rows["target_labels"], dtype=np.int8),
        "competing_labels": np.asarray(rows["competing_labels"], dtype=np.int8),
        "target_write_indices": np.asarray(
            rows["target_write_indices"], dtype=np.int16
        ),
        "competing_write_indices": np.asarray(
            rows["competing_write_indices"], dtype=np.int16
        ),
        "labels_by_context": np.stack(labels_by_context).astype(np.int8),
        "target_colors": presenter.colors.astype(np.float32),
        "waypoint_cells": waypoint_cells,
        "waypoint_centers": waypoint_centers,
        "phase_action_counts": np.asarray(
            phase_action_counts,
            dtype=np.int16,
        ),
    }
    query_indices = np.flatnonzero(arrays["query_mask"])
    if arrays["write_mask"].sum() != 8 or query_indices.size != 4:
        raise RuntimeError("episode write/query count violates protocol")
    if np.any(arrays["visible_target_labels"][query_indices] >= 0):
        raise RuntimeError("a query frame contains a presented target")
    if np.any(arrays["target_geom_radius"][query_indices] > 0.0011):
        raise RuntimeError("a query frame contains a non-hidden target geom")
    if np.any(arrays["target_write_indices"][query_indices] >= query_indices):
        raise RuntimeError("a query target is not sourced from strict history")

    event_positions = _event_pose_table(arrays, "agent_pos")
    waypoint_errors = np.linalg.norm(
        event_positions - waypoint_centers[None],
        axis=-1,
    )
    revisit_errors = np.linalg.norm(
        event_positions - event_positions[0:1],
        axis=-1,
    )
    action_pose = _integrate_action_pose(arrays["actions"])
    arrays_with_action_pose = {**arrays, "action_pose": action_pose}
    event_action_pose = _event_pose_table(
        arrays_with_action_pose,
        "action_pose",
    )
    action_revisit_position = np.linalg.norm(
        event_action_pose[..., :2] - event_action_pose[0:1, ..., :2],
        axis=-1,
    )
    action_revisit_heading = np.abs(
        np.angle(
            np.exp(
                1j
                * (
                    event_action_pose[..., 2]
                    - event_action_pose[0:1, ..., 2]
                )
            )
        )
    )
    maximum_waypoint_error = float(waypoint_errors.max())
    maximum_revisit_error = float(revisit_errors.max())
    if maximum_waypoint_error > MOVE_FINAL_ERROR_LIMIT:
        raise RuntimeError("event waypoint error violates the protocol")
    if maximum_revisit_error > REVISIT_ERROR_LIMIT:
        raise RuntimeError("same-waypoint physical revisit error violates protocol")

    metadata = {
        "environment_seed": environment_seed,
        "layout_sha256": _layout_hash(time_step.observation),
        "route_family_ids": [routes[0].family_id, routes[1].family_id],
        "route_family_hashes": [
            routes[0].canonical_hash,
            routes[1].canonical_hash,
        ],
        "physical_route_sha256": physical_route_hash,
        "cue_mode": "visual_history",
        "cue_label_mapping": cue_label_mapping.tolist(),
        "assignment_sha256": assignment_hash,
        "context_sequence": list(context_sequence),
        "waypoint_cells": waypoint_cells.tolist(),
        "waypoint_centers": waypoint_centers.tolist(),
        "cumulative_outbound_path_m": float(N_WAYPOINTS - 1),
        "minimum_visible_target_score": int(minimum_observed_visibility),
        "maximum_waypoint_center_error_m": maximum_waypoint_error,
        "maximum_same_waypoint_revisit_error_m": maximum_revisit_error,
        "phase_action_counts_before_padding": phase_action_counts,
        "phase_closure_errors_m": phase_closure_errors,
        "phase_heading_errors_rad": phase_heading_errors,
        "maximum_action_ec_revisit_position_drift": float(
            action_revisit_position.max()
        ),
        "maximum_action_ec_revisit_heading_drift_rad": float(
            action_revisit_heading.max()
        ),
        "mean_position_drift": float(
            np.linalg.norm(arrays["agent_pos"][-1] - initial_position)
        ),
        "write_count": int(arrays["write_mask"].sum()),
        "query_count": int(arrays["query_mask"].sum()),
        "delayed_query_count": int(arrays["delayed_query_mask"].sum()),
    }
    return arrays, metadata


def _canonical_assignment_sha256(labels_by_context: np.ndarray) -> str:
    labels = np.asarray(labels_by_context, dtype=np.uint8)
    if labels.shape != (2, N_WAYPOINTS):
        raise ValueError(
            "labels_by_context must have shape [2, n_waypoints]"
        )
    first = tuple(int(value) for value in labels[0].tolist())
    second = tuple(int(value) for value in labels[1].tolist())
    canonical = min((first, second), (second, first))
    return _sha256_bytes(
        bytes((*canonical[0], 255, *canonical[1]))
    )[:16]


def _load_forbidden_manifest_hashes(
    paths: list[Path],
) -> dict[str, set[str]]:
    hashes = {
        "layouts": set(),
        "physical_routes": set(),
        "route_families": set(),
    }
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for episode in payload.get("episodes", []):
            hashes["layouts"].add(str(episode["layout_sha256"]))
            hashes["physical_routes"].add(
                str(episode["physical_route_sha256"])
            )
            hashes["route_families"].update(
                str(value)
                for value in episode["route_family_hashes"]
            )
    return hashes


def _load_forbidden_assignments(paths: list[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as source:
            labels = source["labels_by_context"]
            hashes.update(
                _canonical_assignment_sha256(episode)
                for episode in labels
            )
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate translated simulator waypoint A-B-A data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/memorymaze3d_simulator_translated_waypoint_aba_dev_v1"
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "stage_gate", "test"),
        required=True,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        required=True,
        help="Unique simulator environments; each emits two counterfactual members.",
    )
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--minimum-visibility", type=int, default=80)
    parser.add_argument("--max-attempts-per-episode", type=int, default=16)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--exclude-data",
        type=Path,
        action="append",
        default=[],
        help=(
            "NPZ files whose canonical context-by-waypoint assignments "
            "must not reappear."
        ),
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    output_path = args.output_dir / f"{args.split}.npz"
    manifest_path = args.output_dir / f"manifest_{args.split}.json"
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite {output_path} or {manifest_path}"
        )

    forbidden = _load_forbidden_manifest_hashes(args.exclude_manifest)
    accepted_layouts = set(forbidden["layouts"])
    accepted_physical_routes = set(forbidden["physical_routes"])
    forbidden_route_families = set(forbidden["route_families"])
    accepted_assignments = _load_forbidden_assignments(args.exclude_data)
    episode_arrays: list[dict[str, np.ndarray]] = []
    episode_records = []
    pair_action_mismatches = 0
    pair_query_pixel_mismatches = 0
    pair_query_pose_max_difference = 0.0
    candidate_seed = args.seed_start
    for environment_index in range(args.episodes):
        last_error: Exception | None = None
        for _ in range(args.max_attempts_per_episode):
            try:
                members = [
                    generate_episode(
                        environment_seed=candidate_seed,
                        split=args.split,
                        route_rng_seed=candidate_seed ^ 0xA6B4C2,
                        minimum_visibility=args.minimum_visibility,
                        first_context=first_context,
                    )
                    for first_context in (0, 1)
                ]
                layout_hashes = {
                    metadata["layout_sha256"] for _, metadata in members
                }
                if len(layout_hashes) != 1:
                    raise RuntimeError(
                        "counterfactual members do not share one layout"
                    )
                layout_hash = next(iter(layout_hashes))
                if layout_hash in accepted_layouts:
                    raise RuntimeError("layout hash is not split-unique")
                physical_route_hash = str(
                    members[0][1]["physical_route_sha256"]
                )
                if physical_route_hash in accepted_physical_routes:
                    raise RuntimeError(
                        "physical waypoint route is not split-unique"
                    )
                left, right = members[0][0], members[1][0]
                canonical_assignment_hash = (
                    _canonical_assignment_sha256(
                        left["labels_by_context"]
                    )
                )
                if canonical_assignment_hash in accepted_assignments:
                    raise RuntimeError(
                        "canonical object assignment is not split-unique"
                    )
                route_family_hashes = {
                    str(value)
                    for value in members[0][1]["route_family_hashes"]
                }
                if route_family_hashes & forbidden_route_families:
                    raise RuntimeError(
                        "context route family overlaps an excluded split"
                    )
                if not np.array_equal(left["query_mask"], right["query_mask"]):
                    raise RuntimeError(
                        "counterfactual query masks do not align"
                    )
                action_mismatch = int(
                    np.count_nonzero(left["actions"] != right["actions"])
                )
                query_mask = left["query_mask"]
                pixel_mismatch = int(
                    np.count_nonzero(
                        left["images"][query_mask]
                        != right["images"][query_mask]
                    )
                )
                query_pose_difference = float(
                    max(
                        np.max(
                            np.abs(
                                left["agent_pos"][query_mask]
                                - right["agent_pos"][query_mask]
                            )
                        ),
                        np.max(
                            np.abs(
                                left["agent_dir"][query_mask]
                                - right["agent_dir"][query_mask]
                            )
                        ),
                    )
                )
                if action_mismatch != 0:
                    raise RuntimeError(
                        "counterfactual physical actions differ: "
                        f"{action_mismatch}"
                    )
                if query_pose_difference > 1e-5:
                    raise RuntimeError(
                        "counterfactual query pose differs: "
                        f"{query_pose_difference}"
                    )
                if pixel_mismatch != 0:
                    raise RuntimeError(
                        "counterfactual query RGB differs: "
                        f"{pixel_mismatch}"
                    )
                break
            except RuntimeError as error:
                last_error = error
                candidate_seed += 1
        else:
            raise RuntimeError(
                f"failed to generate environment {environment_index}: "
                f"{last_error}"
            )
        accepted_layouts.add(str(layout_hash))
        accepted_physical_routes.add(physical_route_hash)
        accepted_assignments.add(canonical_assignment_hash)
        pair_action_mismatches += action_mismatch
        pair_query_pixel_mismatches += pixel_mismatch
        pair_query_pose_max_difference = max(
            pair_query_pose_max_difference,
            query_pose_difference,
        )
        for member_index, (arrays, metadata) in enumerate(members):
            metadata["canonical_assignment_sha256"] = (
                canonical_assignment_hash
            )
            episode_arrays.append(arrays)
            episode_records.append(
                {
                    "index": len(episode_records),
                    "pair_id": environment_index,
                    "pair_member": member_index,
                    **metadata,
                }
            )
        print(
            f"[{environment_index + 1:03d}/{args.episodes:03d}] "
            f"split={args.split} seed={candidate_seed} "
            f"visibility={min(row[1]['minimum_visible_target_score'] for row in members)} "
            f"waypoint={max(row[1]['maximum_waypoint_center_error_m'] for row in members):.3f} "
            f"revisit={max(row[1]['maximum_same_waypoint_revisit_error_m'] for row in members):.3f} "
            f"actions={max(max(row[1]['phase_action_counts_before_padding']) for row in members)}",
            flush=True,
        )
        candidate_seed += 1

    payload = {
        key: np.stack([episode[key] for episode in episode_arrays])
        for key in episode_arrays[0]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    manifest = {
        "format": (
            "remap_former.memorymaze3d_simulator_translated_waypoint_aba.v1"
        ),
        "split": args.split,
        "path": output_path.as_posix(),
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "environment": {
            "name": "MemoryMaze3D variable 9x9",
            "camera_resolution": CAMERA_RESOLUTION,
            "good_visibility": True,
            "target_count": TARGET_COUNT,
            "waypoints": N_WAYPOINTS,
            "cumulative_outbound_path_m": float(N_WAYPOINTS - 1),
        },
        "simulator_contract": {
            "actions_executed": int(payload["actions"].size),
            "frames_observed": int(
                payload["images"].shape[0] * payload["images"].shape[1]
            ),
            "post_action_frame_for_every_action": True,
            "teleports_after_reset": 0,
            "episode_steps": EPISODE_STEPS,
            "phase_steps": PHASE_STEPS,
            "visual_history_context": True,
            "paired_counterfactuals": True,
            "generator_pose_controller": True,
            "simulator_pose_model_input": False,
        },
        "model_input_fields": ["actions", "DINO(frame images)"],
        "metadata_only_fields": [
            "agent_pos",
            "agent_dir",
            "context_ids",
            "place_ids",
            "segment_ids",
            "route_family_ids",
            "target_labels",
            "target_write_indices",
            "waypoint_cells",
            "waypoint_centers",
            "phase_action_counts",
        ],
        "episodes": episode_records,
        "aggregate": {
            "write_count": int(payload["write_mask"].sum()),
            "query_count": int(payload["query_mask"].sum()),
            "delayed_query_count": int(payload["delayed_query_mask"].sum()),
            "minimum_visible_target_score": int(
                payload["visibility_scores"][
                    payload["visible_target_labels"] >= 0
                ].min()
            ),
            "query_visible_target_count": int(
                (
                    payload["visible_target_labels"][payload["query_mask"]]
                    >= 0
                ).sum()
            ),
            "query_nonhidden_geom_count": int(
                (
                    payload["target_geom_radius"][payload["query_mask"]]
                    > 0.0011
                ).sum()
            ),
            "maximum_waypoint_center_error_m": float(
                max(
                    episode["maximum_waypoint_center_error_m"]
                    for episode in episode_records
                )
            ),
            "maximum_same_waypoint_revisit_error_m": float(
                max(
                    episode["maximum_same_waypoint_revisit_error_m"]
                    for episode in episode_records
                )
            ),
            "maximum_action_ec_revisit_position_drift": float(
                max(
                    episode["maximum_action_ec_revisit_position_drift"]
                    for episode in episode_records
                )
            ),
            "maximum_action_ec_revisit_heading_drift_rad": float(
                max(
                    episode["maximum_action_ec_revisit_heading_drift_rad"]
                    for episode in episode_records
                )
            ),
            "maximum_phase_actions_before_padding": int(
                max(
                    max(episode["phase_action_counts_before_padding"])
                    for episode in episode_records
                )
            ),
            "unique_layout_count": args.episodes,
            "unique_physical_route_count": len(
                {
                    episode["physical_route_sha256"]
                    for episode in episode_records
                }
            ),
            "unique_canonical_assignment_count": len(
                {
                    episode["canonical_assignment_sha256"]
                    for episode in episode_records
                }
            ),
            "counterfactual_pair_count": args.episodes,
            "counterfactual_action_mismatch_count": pair_action_mismatches,
            "counterfactual_query_pose_max_abs_difference": (
                pair_query_pose_max_difference
            ),
            "counterfactual_query_pixel_mismatch_count": (
                pair_query_pixel_mismatches
            ),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest["aggregate"], indent=2))


if __name__ == "__main__":
    main()
